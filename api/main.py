from pathlib import Path
import io
import zipfile

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.zip"

app = FastAPI(
    title="Wine Multiclass Prediction API",
    description="Three-class wine classification using a model trained in Databricks and tracked with MLflow.",
    version="1.0.0",
)

model = None
feature_names = []
class_names = []


@app.on_event("startup")
def load_model():
    global model, feature_names, class_names

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model archive not found at: {MODEL_PATH}. "
            "Copy the model.zip downloaded from your Databricks Volume into the model folder."
        )

    try:
        with zipfile.ZipFile(MODEL_PATH, "r") as archive:
            if "model.joblib" not in archive.namelist():
                raise RuntimeError(
                    "model.zip must contain a file named model.joblib at its root."
                )
            with archive.open("model.joblib") as model_file:
                bundle = joblib.load(io.BytesIO(model_file.read()))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("model.zip is not a valid ZIP archive.") from exc

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]
    print(f"Model loaded. Features: {len(feature_names)}; classes: {class_names}")


class WineFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float = Field(alias="od280/od315_of_diluted_wines")
    proline: float


@app.get("/")
def home():
    return {
        "message": "Wine Multiclass Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "model_type": type(model).__name__,
        "features": feature_names,
        "classes": class_names,
    }


@app.post("/predict")
def predict(data: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    payload = data.model_dump(by_alias=True)
    row = [payload[feature] for feature in feature_names]
    input_array = np.asarray([row], dtype=float)

    predicted_id = int(model.predict(input_array)[0])
    probabilities = model.predict_proba(input_array)[0]

    return {
        "predicted_class_id": predicted_id,
        "predicted_class": class_names[predicted_id],
        "probabilities": {
            class_names[index]: round(float(probability), 4)
            for index, probability in enumerate(probabilities)
        },
    }
