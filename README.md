# 📈 Stock ETL and Prediction Pipeline

This project is an end-to-end stock market prediction system that automates data extraction, transformation, model training, and prediction delivery. It is powered by AlphaVantage, FastAPI, MLflow, Docker, and GitHub Actions — and is **live on an AWS EC2 instance**.

---

## 🌐 Live Demo

**🔗 API Base URL:** `http://13.60.179.140/`
**📘 Swagger Docs:** `http://13.60.179.140/docs`

---

## 🚀 Key Features

* ✅ **ETL pipeline** to fetch and process stock data via AlphaVantage
* ✅ **170+ technical indicators** with statistical feature selection (41 selected)
* ✅ **MLPClassifier model** for binary stock movement prediction
* ✅ **FastAPI** server for real-time prediction and historical data access
* ✅ **MLflow** for experiment tracking and model versioning
* ✅ **Dockerized** for easy deployment
* ✅ **CI/CD** with GitHub Actions
* ✅ **Deployed** and running on AWS EC2

---

## 🗂️ Project Structure

```
stock-etl-pipeline/
│
├── app/                  # FastAPI app logic
│   ├── main.py
│   ├── models/  #Saved Models
│   ├── mlruns/  # MLflow tracking
│   ├── data/
│   │    ├── raw/ #Raw Dataset
│   │    ├── processed/ #Dataset after feature engineering 
│   │    ├── results/ # Model Training Results 
│   │     
│   └── scripts/
│        ├──api_data_fetcher.py/                  # ETL scripts
│        ├── config.py
│        ├── database_handler.py
│        ├── feature_engineering.py
│        ├── model_trainer.py
│        └── statistical_measures.py
│
├── .github/workflows/    # CI/CD configs
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Development (Optional)

1. **Clone the repo**

```bash
git clone https://github.com/your-username/stock-etl-pipeline.git
cd stock-etl-pipeline
```

2. **Set environment variable**

Create a `v.env` file with your API key from AlphaVantage:

```
API_KEY="your_key_here"
DB_NAME="data.sqlite"
MODEL_DIR="models"
```

3. **Build and run Docker containers**

```bash
docker-compose up --build
```

4. **Access API locally**

Go to [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧠 Model Details

* **Type**: MLPClassifier (Multi-layer perceptron)
* **Input**: 41 selected technical indicators
* **Output**: Up or Down stock movement
* **Training Pipeline**: Includes feature scaling, selection, and resampling (if needed)
* **Tracking**: Experiments tracked using MLflow

---

## 🔄 CI/CD Workflow

* GitHub Actions runs linting, testing, and Docker build on push
* Deployment handled via SSH to EC2 or Docker Compose on server
* ML models are versioned and logged via MLflow

---

## 📦 API Endpoints

* `POST /historical_prediction/` – Make Predictions Based on Existing Data
* `POST /predict_stock_directions/` – Get Prediction for Input Features
* `POST /load_data_from_api/` – Get data from AlphaVantage
* `GET /docs` – Interactive API Documentation (Swagger UI)
* `GET /model_accuracy` – Get Model Accuracy

---

## 🛠 Technologies Used

* Python · FastAPI · Scikit-learn · MLflow · Docker · GitHub Actions · AWS EC2 · AlphaVantage API

---

## ✍️ Author

**Ridwan Yusuf: Data Scientist | Exploring MLOps**

[Medium Articles](https://medium.com/@gentroyal) | [LinkedIn](https://www.linkedin.com/in/yusufridwan/)