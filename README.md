# Wine Multiclass FastAPI + Docker (VS Code)

This is the local inference/deployment project for the multiclass model trained and tracked in Databricks MLflow.

## Important: add your trained model

The ZIP does not contain your private Databricks-trained model because that file must be downloaded from your Databricks workspace.

1. Download `model.zip` from:
   `/Volumes/multidemoone/multiwinedemo/onewindemo/model/model.zip`
2. Copy it into this project's `model` folder.
3. The final path must be:
   `model/model.zip`
4. Do not extract `model.zip`.

The archive must contain `model.joblib` at its root. That file should contain the dictionary keys `model`, `feature_names`, and `class_names`, as created by the Databricks notebook.

## Project structure

```text
wine_multiclass_fastapi_vscode/
├── api/
│   ├── main.py
│   └── requirements.txt
├── model/
│   └── model.zip                 # Copy this from Databricks
├── .vscode/
│   └── launch.json
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## A. Run in VS Code on Windows

Open this folder in VS Code. Open Terminal > New Terminal and make sure the terminal is in the project root.

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 2. Activate it

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r api/requirements.txt
```

### 4. Start FastAPI

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

Model information:

http://127.0.0.1:8000/model-info

## B. Test POST /predict

In Swagger UI, open `POST /predict`, click **Try it out**, and submit:

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280/od315_of_diluted_wines": 3.4,
  "proline": 1050
}
```

The response includes the predicted class ID, class name, and probability for each class.

## C. Build and run with Docker

Docker Desktop must be running. Ensure `model/model.zip` exists before building.

```powershell
docker build -t wine-multiclass-api:1.0 .
```

Run:

```powershell
docker run --name wine-api -p 8000:8000 wine-multiclass-api:1.0
```

Open:

http://127.0.0.1:8000/docs

Stop and remove:

```powershell
docker stop wine-api
docker rm wine-api
```

## Troubleshooting

- **Model archive not found:** check that `model/model.zip` exists in the project root.
- **Invalid ZIP:** make sure you downloaded the ZIP produced by the Databricks notebook, not a renamed `.joblib` file.
- **Missing `model.joblib` inside ZIP:** open the archive and confirm `model.joblib` is at the ZIP root, not inside an extra nested folder.
- **Feature key error:** the model bundle's `feature_names` must match the 13 Wine dataset feature names.
- **Port in use:** run Uvicorn on port 8001 or Docker with `-p 8001:8000`.

## End-to-end flow

Databricks training -> MLflow experiment/model logging -> export `model.zip` to Unity Catalog Volume -> download `model.zip` -> FastAPI inference -> Docker container.
