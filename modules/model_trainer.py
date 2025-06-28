import pandas as pd
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


def detect_problem_type(y_series):
    if y_series.dtype == 'object':
        return "Classification"
    elif y_series.nunique() < 20:
        return "Classification"
    else:
        return "Regression"


def get_models(task_type):
    if task_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Neural Network (MLP)": MLPClassifier(max_iter=1000)
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Support Vector Regressor": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Neural Network (MLP)": MLPRegressor(max_iter=1000)
        }

def select_best_model_name(evaluation, task_type):
    metric_key = "Accuracy" if task_type == "Classification" else "R2 Score"
    return max(
        (m for m in evaluation if metric_key in evaluation[m]),
        key=lambda m: evaluation[m][metric_key]
    )


def train_model(df: pd.DataFrame, target_column: str):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    task_type = detect_problem_type(y)

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models(task_type)
    evaluation_results = {}
    trained_models = {}
    best_model = None
    best_score = 0

    for name, estimator in models.items():
        model = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator)
        ])

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = model

            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                evaluation_results[name] = {
                    "Accuracy": round(acc, 4)
                }
                if acc > best_score:
                    best_score = acc
                    best_model = model
            else:
                r2 = r2_score(y_test, y_pred)
                evaluation_results[name] = {
                    "R2 Score": round(r2, 4)
                }
                if r2 > best_score:
                    best_score = r2
                    best_model = model
        except Exception as e:
            evaluation_results[name] = {"Error": str(e)}

    return {
        "evaluation": evaluation_results,
        "task_type":task_type,
        "best_model":best_model
    }


def get_serialized_model(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return buffer
