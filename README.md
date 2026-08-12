# Loan Approval Predictor — Streamlit App

A deployable web app version of the loan approval notebook. Applicants'
details go in through a form; a trained Random Forest model returns an
Approved/Rejected prediction with a confidence score.

## Project structure

```
loan_app/
├── app.py                  # Streamlit UI
├── train_model.py          # Reproduces cleaning/encoding/scaling + trains the model
├── requirements.txt
├── loan_approval_dataset.csv   # <- put the dataset here before training
└── artifacts/               # created by train_model.py
    ├── model.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── feature_columns.pkl
    ├── scale_columns.pkl
    └── metrics.json
```

## What changed vs. the notebook

The notebook trained several models (Logistic Regression, Decision Tree,
Random Forest, KNN, SVM, AdaBoost, XGBoost) and saved the raw Random Forest
model with `joblib.dump(rf, 'Loan_model.pkl')`. Two things were missing for
a real deployment:

1. **The `StandardScaler` and label encoders weren't saved.** New user
   input has to go through *identical* scaling/encoding before the model
   sees it, or predictions will be wrong. `train_model.py` now saves the
   scaler and encoders alongside the model.
2. **`loan_id` was left in as a feature.** It's a row identifier, not a
   predictor — the notebook only dropped `loan_status` before splitting
   into `x`/`y`, so `loan_id` sneaked into training. `train_model.py`
   drops it explicitly so the model isn't keying off a meaningless number.

## 1. Train the model

```bash
cd loan_app
pip install -r requirements.txt
# put loan_approval_dataset.csv in this folder
python train_model.py
```

This writes the `artifacts/` folder the app needs. Check
`artifacts/metrics.json` for the test accuracy and classification report.

## 2. Run the app locally

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 3. Deploy it for free — Streamlit Community Cloud

1. Push this folder to a GitHub repo (include `artifacts/`, or set up a
   build step that runs `train_model.py` — simplest is to commit the
   artifacts directly since they're small `.pkl` files).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click **New app**, pick the repo/branch, set main file path to `app.py`.
4. Deploy. You'll get a public `https://<your-app>.streamlit.app` URL.

### Alternative: deploy with Docker (any cloud VM / Render / Railway / Fly.io)

Add this `Dockerfile` to the folder:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then:
```bash
docker build -t loan-approval-app .
docker run -p 8501:8501 loan-approval-app
```

Push the image to any container host (Render, Railway, Fly.io, AWS
ECS/App Runner, GCP Cloud Run) and point it at port 8501.

## Notes

- If you retrain on updated data, just rerun `train_model.py` — the app
  picks up new artifacts automatically (restart the app / clear cache).
- `education` and `self_employed` must exactly match the categories seen
  at training time (`Graduate`/`Not Graduate`, `Yes`/`No`); the dropdowns
  already enforce this.
