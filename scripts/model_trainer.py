import os
import numpy as np
import time
from datetime import datetime
import pandas as pd
import pickle as pkl

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from statistical_measures import Statistics
        
def ensure_data_dir():
    df_path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'data', 'processed'))
    os.makedirs(df_path, exist_ok = True)

    result_path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'data', 'results'))
    os.makedirs(result_path, exist_ok = True)

    model_path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'models'))
    os.makedirs(model_path, exist_ok = True)
    
    return df_path, result_path, model_path

def vertical_split(table_name, target):
    
    df_path, result_path, model_path = ensure_data_dir()
    df = pd.read_csv(f'{df_path}/{table_name}_indicators.csv')
    
    X = df.drop(columns = [target])
    X = (X - X.min())/(X.max() - X.min())
    y = df[target]

    return X, y, result_path, model_path

def train_with_kfold(table_name, target): 
    
    X, y, result_path, model_path = vertical_split(table_name, target)
    
    sta = Statistics()
    statistical_methods = {
        'PCA': sta.pca_ranking,
        'Dispersion_Ratio': sta.dispersion_ratio,
        'Chi_Square': sta.chi_square_ranking,
        'Pearson_Correlation': sta.pearson_correlation,
        'Mean_Absolute_Difference': sta.mean_absolute_difference,
        'Low_Variance': sta.low_variance
        }
    
    top_features_all_methods = []
        
    for method_name, method in statistical_methods.items():
        print(f"Applying {method_name} feature selection...")
        ranked_features = method(X, y)
        top_quartile_count = max(1, len(ranked_features) // 4)
        top_features = ranked_features[:top_quartile_count]
        top_features_all_methods.append(set(top_features))
    
    feature_counts = Counter(f for s in top_features_all_methods for f in s)
    
    selected_sets = {
            n: {f for f, count in feature_counts.items() if count >= n}
            for n in range(1, len(statistical_methods) + 1)
        }
    
    results = {}
    models = {}

    model = MLPClassifier(activation = 'logistic', solver = 'lbfgs', batch_size = 'auto', learning_rate = 'adaptive', \
                                    learning_rate_init = 0.03, max_iter = 5000, \
                                    momentum = 0.2, \
                                    random_state = np.random.get_state()[1][0], \
                                    early_stopping = False)

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for n in range(0, len(statistical_methods) + 1):
        if n == 0:
            selected_features = X.columns.tolist()
        else:
            selected_features = list(selected_sets[n])
            if not selected_features:
                results[n] = {
                        'num_features': 0,
                        'features': [],
                        'accuracy': None
                    }
                print(f"No Features Selected for Threshold n = {n}")
                continue
    
        print(f"Evaluating Model on {'All Features' if n == 0 else f'Features Selected by ≥ {n} Methods'}...")
        X_selected = X[selected_features]
    
        accuracies = []
        epochs_used = []
        cv_generator = StratifiedKFold(n_splits = 10, shuffle = False)
        t1 = time.time()
    
        for train_idx, test_idx in cv_generator.split(X_selected, y):
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
            hidden_layer_sizes = int((X_train.shape[1] + len(np.unique(y_train))) / 2)
            model.hidden_layer_sizes = hidden_layer_sizes
    
            model.fit(X_train, y_train)

            import mlflow
            import mlflow.sklearn
            from mlflow.models.signature import infer_signature
                
            mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use your server address
            mlflow.set_experiment("Feature_Selection_Experiments")

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                    sk_model = model,
                    artifact_path = "model",
                    input_example = X_train[:5],
                    signature = signature
                )

            y_pred = model.predict(X_test)
    
            accuracies.append(accuracy_score(y_test, y_pred))
            epochs_used.append(model.n_iter_)
    
        duration = time.time() - t1
        avg_accuracy = np.median(accuracies)
        avg_epochs = int(np.median(epochs_used))
    
        results[n] = {
                'num_features': len(selected_features),
                'features': selected_features,
                'accuracy': avg_accuracy,
                'time_taken': duration,
                'epochs': avg_epochs
            }
    
        models[n] = model
    
        print(f"Threshold n = {n}: {len(selected_features)} features, accuracy = {avg_accuracy:.4f}")
    
        # Log to MLflow
        with mlflow.start_run(run_name = f"KFold_n_{n}_{timestamp}", nested = True):
            mlflow.log_param("threshold_n", n)
            mlflow.log_param("num_features", len(selected_features))
            mlflow.log_metric("accuracy", avg_accuracy)
            mlflow.log_metric("training_time_sec", duration)
            mlflow.log_metric("epochs", avg_epochs)
    
            mlflow.set_tag("developer", "Ridwan")
    
            # Log the model
            mlflow.sklearn.log_model(model, artifact_path = "model", input_example = X_train[:5])
    
    result_df = pd.DataFrame(results).T
    result_df = result_df.query('num_features > 0').reset_index(drop=True)
    result_csv_path = f"{result_path}/results_{timestamp}.csv"
    result_df.to_csv(result_csv_path, index=False)
    
    # Log CSV once (not for each run)
    with mlflow.start_run(run_name = f"Final_Results_{timestamp}", nested = True):
        mlflow.log_artifact(result_csv_path, artifact_path="results")
    
    best_n = max(
            (k for k in results if results[k]['accuracy'] is not None),
            key=lambda x: results[x]['accuracy']
        )
    
    best_model = models[best_n]
    model_file_path = f"{model_path}/model_{timestamp}.pkl"
    
    with open(model_file_path, 'wb') as f:
        pkl.dump(best_model, f)
    
    # Log final best model separately
    with mlflow.start_run(run_name = f"Best_Model_n_{best_n}_{timestamp}", nested = True):
        mlflow.log_param("best_n", best_n)
        mlflow.log_metric("best_accuracy", results[best_n]["accuracy"])
        mlflow.log_artifact(model_file_path, artifact_path="best_model")
    
    return result_df
