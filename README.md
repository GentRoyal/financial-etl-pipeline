# 📈 Stock ETL and Prediction Pipeline

This project is an end-to-end stock market prediction system that automates data extraction, transformation, model training, and prediction delivery. It is powered by AlphaVantage, FastAPI, MLflow, Docker, and GitHub Actions — and is **live on an AWS EC2 instance**.

---

## 🌐 Live Demo

**🔗 API Base URL:** `http://<your-ec2-public-ip>:8000`
**📘 Swagger Docs:** `http://<your-ec2-public-ip>:8000/docs`

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
│   ├── routes/
│   └── utils/
│
├── etl/                  # ETL scripts
│   ├── extract.py
│   ├── transform.py
│   └── load.py
│
├── models/               # Saved model files
├── data/                 # Processed data (local testing)
├── mlflow/               # MLflow tracking
├── .github/workflows/    # CI/CD configs
├── Dockerfile
├── docker-compose.yml
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

Create a `.env` file with your API key:

```
ALPHAVANTAGE_API_KEY=your_key_here
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

* `GET /historical-data/` – View historical data for a stock
* `POST /predict/` – Get prediction for input features
* `GET /docs` – Interactive API documentation (Swagger UI)

---

## 🛠 Technologies Used

* Python · FastAPI · Scikit-learn · MLflow · Docker · GitHub Actions · AWS EC2 · AlphaVantage API

---

## ✍️ Author

**Ridwan Yusuf**
*Data Scientist | Exploring MLOps
[Medium Articles](https://medium.com/@gentroyal) | [LinkedIn](https://www.linkedin.com/in/yusufridwan/)
