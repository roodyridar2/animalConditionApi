# main.py
import io
import base64
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

# CORS
from fastapi.middleware.cors import CORSMiddleware

# ML / Data libs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# Matplotlib non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import uvicorn

########################################
# FastAPI Initialization
########################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
# Global State
########################################
df = None  # Full dataframe
X_train_sm = None
y_train_sm = None
X_test = None
y_test = None
results = {}
feature_selector = None

# NEW global dictionaries for storing model & label encoders
trained_models = {}
label_encoders = {}   # key = column name, value = fitted LabelEncoder

########################################
# Pydantic Models
########################################
class TrainRequest(BaseModel):
    feature_selection_method: str  # "SelectKBest" or "RFE"
    model_name: str  # "All" or one of "NaiveBayes"/"kNN"/"SVM"/"NeuralNet"
    hyperparams: Optional[Dict[str, Any]] = None
    # e.g. { "knn_neighbors": 7, "svm_c": 0.5, "nn_epochs": 200 }

# For real-time inference
class PredictRequest(BaseModel):
    AnimalName: str
    symptoms1: str
    symptoms2: str
    symptoms3: str
    symptoms4: str
    symptoms5: str
    model_name: str = "All"  # or "NaiveBayes", "kNN", "SVM", "NeuralNet"


########################################
# Endpoints
########################################

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    1. Upload CSV.
    2. Preprocess data (example steps).
    3. Apply label encoding to all features.
    4. Split into train/test.
    5. Apply SMOTE.
    6. Store in global variables.
    """
    global df, X_train_sm, y_train_sm, X_test, y_test, label_encoders

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df_local = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")

    try:
        # Example pre-processing
        symptom_cols = ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]
        df_local[symptom_cols] = df_local[symptom_cols].apply(lambda x: x.str.lower().str.strip())

        # Make sure target column "Dangerous" isn't NaN
        df_local.dropna(subset=["Dangerous"], inplace=True)

        # X / y
        feature_cols = ["AnimalName"] + symptom_cols
        X = df_local[feature_cols].copy()
        y = df_local["Dangerous"].copy()

        # Encode X features with LabelEncoders, store them
        label_encoders = {}
        for col in feature_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Encode target
        # We don't necessarily store the target encoder here because it's just 2 classes
        # but you COULD store it in case you want to decode predictions later.
        y = LabelEncoder().fit_transform(y)

        # Train / test split
        X_train, X_test_split, y_train, y_test_split = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        # SMOTE for class balancing
        sm = SMOTE(random_state=42)
        X_train_sm_local, y_train_sm_local = sm.fit_resample(X_train, y_train)

        # Store in globals
        df = df_local
        X_train_sm = X_train_sm_local
        y_train_sm = y_train_sm_local
        X_test = X_test_split
        y_test = y_test_split

        return {"detail": "Dataset uploaded and preprocessed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {e}")


@app.get("/dataset-preview")
def get_dataset_preview():
    """
    Returns the entire DataFrame (converted to JSON) along with column names.
    """
    global df
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload first.")
    preview_df = df.copy()
    return {
        "preview_data": preview_df.to_dict(orient="records"),
        "columns": preview_df.columns.tolist()
    }


@app.post("/train-models")
def train_models(request: TrainRequest):
    """
    1. Perform feature selection (SelectKBest or RFE).
    2. Train one or multiple models (NaiveBayes, kNN, SVM, NeuralNet).
    3. Use manual hyperparams if provided or GridSearchCV param grids.
    4. Evaluate and store results globally.
    5. Store each trained model in global dictionary `trained_models`.
    """
    global X_train_sm, y_train_sm, X_test, y_test, results, feature_selector, trained_models

    if X_train_sm is None or y_train_sm is None or X_test is None or y_test is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload first.")

    # Feature selection
    if request.feature_selection_method == "SelectKBest":
        selector = SelectKBest(score_func=chi2, k=5)
    elif request.feature_selection_method == "RFE":
        rfe_model = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFE(estimator=rfe_model, n_features_to_select=5)
    else:
        raise HTTPException(status_code=400, detail="Invalid feature selection method.")

    X_train_fs = selector.fit_transform(X_train_sm, y_train_sm)
    X_test_fs = selector.transform(X_test)
    feature_selector = selector

    # Extract optional hyperparams
    hyperparams = request.hyperparams if request.hyperparams else {}

    # Manual hyperparams
    k_neighbors = hyperparams.get("knn_neighbors", 5)
    svm_c = hyperparams.get("svm_c", 1.0)
    nn_epochs = hyperparams.get("nn_epochs", 1000)

    # Build base models
    models = {
        "NaiveBayes": GaussianNB(),
        "kNN": KNeighborsClassifier(n_neighbors=k_neighbors),
        "SVM": SVC(probability=True, random_state=42, C=svm_c),
        "NeuralNet": MLPClassifier(max_iter=nn_epochs, random_state=42),
    }

    # Param grids for GridSearchCV (if user has NOT provided hyperparams):
    param_grids = {
        "kNN": {"n_neighbors": [3, 5, 7, 9]},
        "SVM": {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1], "kernel": ["rbf"]},
        "NeuralNet": {"hidden_layer_sizes": [(16, 8), (32, 16)], "alpha": [0.0001, 0.001]},
    }

    selected_model_name = request.model_name  # "All", "NaiveBayes", "kNN", "SVM", "NeuralNet"

    # Filter models if user wants just one
    if selected_model_name != "All":
        if selected_model_name not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model_name: {selected_model_name}")
        models = {selected_model_name: models[selected_model_name]}

    local_results = {}

    for model_name, model in models.items():
        # Check if user provided manual hyperparams for each model
        user_provided_params = False
        if model_name == "kNN" and "knn_neighbors" in hyperparams:
            user_provided_params = True
        elif model_name == "SVM" and "svm_c" in hyperparams:
            user_provided_params = True
        elif model_name == "NeuralNet" and "nn_epochs" in hyperparams:
            user_provided_params = True

        if model_name in param_grids and not user_provided_params:
            # Use GridSearchCV
            grid_search = GridSearchCV(model, param_grids[model_name], scoring="accuracy", cv=3)
            grid_search.fit(X_train_fs, y_train_sm)
            best_model = grid_search.best_estimator_
        else:
            # Use the manual hyperparams
            model.fit(X_train_fs, y_train_sm)
            best_model = model

        # Store the trained model
        trained_models[model_name] = best_model

        # Evaluate
        y_pred = best_model.predict(X_test_fs)
        y_proba = best_model.predict_proba(X_test_fs)[:, 1] if hasattr(best_model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        conf_mat = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(
            y_test, y_pred, target_names=["No", "Yes"], output_dict=True
        )

        local_results[model_name] = {
            "accuracy": acc,
            "f1_score": f1,
            "auc": auc_val,
            "conf_matrix": conf_mat,
            "classification_report": class_report,
            "proba": y_proba.tolist() if y_proba is not None else None,
        }

    results = local_results
    return {"detail": "Models trained successfully."}


@app.get("/results")
def get_results():
    """
    Returns the global 'results' dictionary and indicates which model is best by highest accuracy.
    """
    global results

    if not results:
        raise HTTPException(status_code=400, detail="No results found. Please train the models first.")

    # If there's only one model in results, that's automatically the "best"
    if len(results) == 1:
        only_model_name = list(results.keys())[0]
        best_model_info = results[only_model_name]
        reason = f"Only {only_model_name} was trained. Accuracy: {best_model_info['accuracy']:.3f}."
        if best_model_info["auc"] is not None:
            reason += f" AUC-ROC = {best_model_info['auc']:.3f}."
        return JSONResponse({
            "results": results,
            "best_model": only_model_name,
            "reason": reason
        })

    # Otherwise, pick best by accuracy among all trained
    best_model_name = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
    best_model_info = results[best_model_name]
    reason = f"It achieved the highest accuracy ({best_model_info['accuracy']:.3f})."
    if best_model_info["auc"] is not None:
        reason += f" AUC-ROC = {best_model_info['auc']:.3f}."

    return JSONResponse({
        "results": results,
        "best_model": best_model_name,
        "reason": reason
    })




########################################
# NEW: Real-time inference endpoint
########################################
@app.post("/predict")
def predict_symptoms(request: PredictRequest):
    """
    Given an animal name and 5 symptoms, predict whether it's "Dangerous" or not,
    using one or all trained models.
    """
    global trained_models, label_encoders, feature_selector

    # 1) Check if we have trained any model:
    if not trained_models:
        raise HTTPException(status_code=400, detail="No trained models found. Please train first.")

    # 2) Convert input to a DataFrame
    input_dict = {
        "AnimalName": [request.AnimalName],
        "symptoms1": [request.symptoms1],
        "symptoms2": [request.symptoms2],
        "symptoms3": [request.symptoms3],
        "symptoms4": [request.symptoms4],
        "symptoms5": [request.symptoms5],
    }
    input_df = pd.DataFrame(input_dict)
    for col in input_df.columns:
        # Make sure columns are strings first except for the first column
        if col != "AnimalName":
            input_df[col] = input_df[col].astype(str).str.lower().str.strip()


    # 3) Apply the same label encoding used during training
    for col in input_df.columns:
        if col in label_encoders:
            # Transform each column with the stored label encoder
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
        else:
            raise HTTPException(status_code=400, detail=f"No label encoder found for column '{col}'.")

    # 4) Apply the feature selection transform if applicable
    if feature_selector:
        input_fs = feature_selector.transform(input_df)
    else:
        input_fs = input_df  # or raise an error if needed

    # 5) Predict with either a single model or multiple
    predictions = {}
    if request.model_name != "All":
        if request.model_name not in trained_models:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not found.")
        model = trained_models[request.model_name]
        y_pred = model.predict(input_fs)
        y_proba = model.predict_proba(input_fs)[:, 1] if hasattr(model, "predict_proba") else None

        # Map 0/1 back to "No"/"Yes" if desired
        label_map = {0: "No", 1: "Yes"}
        pred_label = label_map[int(y_pred[0])]

        predictions[request.model_name] = {
            "prediction_raw": int(y_pred[0]),
            "prediction_label": pred_label,
            "probability": float(y_proba[0]) if y_proba is not None else None,
        }
    else:
        # Use all trained models
        for m_name, model in trained_models.items():
            y_pred = model.predict(input_fs)
            y_proba = model.predict_proba(input_fs)[:, 1] if hasattr(model, "predict_proba") else None

            label_map = {0: "No", 1: "Yes"}
            pred_label = label_map[int(y_pred[0])]

            predictions[m_name] = {
                "prediction_raw": int(y_pred[0]),
                "prediction_label": pred_label,
                "probability": float(y_proba[0]) if y_proba is not None else None,
            }

    # 6) Return results
    return {"predictions": predictions}

@app.get("/unique-values")
def get_unique_values():
    """
    Returns the unique values for AnimalName, symptoms1, ..., symptoms5
    in the *already uploaded* dataset.
    """
    global df
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload first.")

    # If your columns are: AnimalName, symptoms1..symptoms5
    unique_animal_names = sorted(df["AnimalName"].dropna().unique().tolist())
    unique_s1 = sorted(df["symptoms1"].dropna().unique().tolist())
    unique_s2 = sorted(df["symptoms2"].dropna().unique().tolist())
    unique_s3 = sorted(df["symptoms3"].dropna().unique().tolist())
    unique_s4 = sorted(df["symptoms4"].dropna().unique().tolist())
    unique_s5 = sorted(df["symptoms5"].dropna().unique().tolist())

    return {
        "animalNames": unique_animal_names,
        "symptoms1": unique_s1,
        "symptoms2": unique_s2,
        "symptoms3": unique_s3,
        "symptoms4": unique_s4,
        "symptoms5": unique_s5,
    }

########################################
# Entry Point
########################################
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
