#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import os
import zipfile
import json
import gzip
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)


# Rutas

TRAIN_ZIP = "files/input/train_data.csv.zip"
TEST_ZIP = "files/input/test_data.csv.zip"
TRAIN_CSV = "files/input/train_data.csv"
TEST_CSV = "files/input/test_data.csv"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"


# PASO 1: Limpieza de datos

def extract_zip_if_needed(zip_path: str, csv_name: str) -> None:
    if not os.path.exists(csv_name):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(csv_name))
            

def load_and_preprocess():

    os.makedirs("files/input", exist_ok=True)

    extract_zip_if_needed(TRAIN_ZIP, TRAIN_CSV)
    extract_zip_if_needed(TEST_ZIP, TEST_CSV)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Crear Age y eliminar columnas
    for df in (train, test):
        df["Age"] = 2021 - df["Year"]
        df.drop(columns=["Year", "Car_Name"], inplace=True)

    train.dropna(inplace=True)
    test.dropna(inplace=True)

    return train, test


# PASO 2: Separar X_train, y_train, X_test, y_test 

def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return X, y


# PASO 3: Pipeline

def make_pipeline(x_train: pd.DataFrame) -> Pipeline:

    categorical = [c for c in x_train.columns if x_train[c].dtype == "object"]
    numeric = [c for c in x_train.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", MinMaxScaler(), numeric),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    return pipeline


# PASO 4: GRIDSEARCHCV

def tune_hyperparameters(pipeline: Pipeline, X_train, y_train) -> GridSearchCV:

    param_grid = {
        "feature_selection__k": ["all", 20, 30, 40, 50],
        "regressor__fit_intercept": [True, False],
    }

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )

    gs.fit(X_train, y_train)
    return gs

# PASO 5: Guardar modelo

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# PASO 6: Calcular métricas y guardarlas

def calculate_metrics(model, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "mad": float(median_absolute_error(y_train, y_train_pred)),
    }

    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred)),
    }

    return train_metrics, test_metrics

def save_metrics(train_metrics, test_metrics):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf8") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")


# Ejecución del proceso

def main():

    train, test = load_and_preprocess()
    X_train, y_train = split_X_y(train)
    X_test, y_test = split_X_y(test)

    pipeline = make_pipeline(X_train)

    model = tune_hyperparameters(pipeline, X_train, y_train)

    save_model(model)

    train_metrics, test_metrics = calculate_metrics(
        model, X_train, y_train, X_test, y_test
    )

    save_metrics(train_metrics, test_metrics)


if __name__ == "__main__":
    main()