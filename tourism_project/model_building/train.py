import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import mlflow
import os

# Define paths
XTRAIN_PATH = "Xtrain.csv"
XTEST_PATH = "Xtest.csv"
YTRAIN_PATH = "ytrain.csv"
YTEST_PATH = "ytest.csv"
MODEL_DEPLOYMENT_PATH = "tourism_project/deployment/model.joblib"
MLFLOW_TRACKING_URI = "http://0.0.0.0:5000" # Local MLflow server, as per workflow step

# Ensure the deployment directory exists
os.makedirs(os.path.dirname(MODEL_DEPLOYMENT_PATH), exist_ok=True)

# Load data
try:
    Xtrain = pd.read_csv(XTRAIN_PATH)
    Xtest = pd.read_csv(XTEST_PATH)
    ytrain = pd.read_csv(YTRAIN_PATH).squeeze() # .squeeze() to convert DataFrame to Series
    ytest = pd.read_csv(YTEST_PATH).squeeze()
    print("Train and test splits loaded successfully.")
except FileNotFoundError:
    print("Error: One or more data split files not found. Please ensure prep.py was run correctly.")
    exit()

# Identify categorical and numerical features
# Based on the data description and typical feature types
numerical_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'
]

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Define the model pipeline
# use_label_encoder=False and eval_metric='logloss' suppress warnings for newer XGBoost versions
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Define hyperparameter grid for GridSearchCV
# The grid is kept relatively small for faster execution in a notebook environment
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__subsample': [0.7, 0.9],
    'classifier__colsample_bytree': [0.7, 0.9]
}

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("tourism_package_prediction")

with mlflow.start_run():
    # Log the full hyperparameter grid being searched
    mlflow.log_params({f"grid_{k}": str(v) for k, v in param_grid.items()})

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3, # Using 3-fold cross-validation for reasonable speed
        scoring='roc_auc', # ROC AUC is suitable for classification tasks, especially with potential class imbalance
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBest parameters found: ", best_params)
    print("Best ROC AUC score from cross-validation: ", grid_search.best_score_)

    # Log best parameters to MLflow
    mlflow.log_params(best_params)

    # Evaluate the best model on the test set
    y_pred = best_model.predict(Xtest)
    y_proba = best_model.predict_proba(Xtest)[:, 1] # Probability of the positive class

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    roc_auc = roc_auc_score(ytest, y_proba)

    print("\nModel Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(ytest, y_pred))

    # Log metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc
    })

    # Save the best model locally for deployment
    joblib.dump(best_model, MODEL_DEPLOYMENT_PATH)
    print(f"\nBest model saved to {MODEL_DEPLOYMENT_PATH}")

    # Log the model to MLflow as an artifact
    # Explicitly trust XGBoost types for logging
    mlflow.sklearn.log_model(best_model, "best_xgboost_model", 
                             skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'])
    print("Model logged to MLflow.")

print("Model training and evaluation complete.")
