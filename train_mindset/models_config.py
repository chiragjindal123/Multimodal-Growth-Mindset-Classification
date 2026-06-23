# models_config.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.base import clone

def get_model_and_space(model_name, trial):
    if model_name == "Logistic Regression":
        C = trial.suggest_loguniform("C", 1e-3, 10)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=C, penalty=penalty)
        steps = [("smote", SMOTE(k_neighbors=1)), ("scaler", StandardScaler()), ("clf", clf)]

    elif model_name == "SVM":
        C = trial.suggest_loguniform("C", 1e-3, 10)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        clf = SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced")
        steps = [("smote", SMOTE(k_neighbors=1)), ("scaler", StandardScaler()), ("clf", clf)]

    elif model_name == "Random Forest":
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
        clf = RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, max_depth=max_depth)
        steps = [("smote", SMOTE(k_neighbors=1)), ("clf", clf)]

    elif model_name == "Gradient Boosting":
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 0.5)
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        steps = [("smote", SMOTE(k_neighbors=1)), ("clf", clf)]

    elif model_name == "Decision Tree":
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, class_weight="balanced")
        steps = [("smote", SMOTE(k_neighbors=1)), ("clf", clf)]

    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 15)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        steps = [("smote", SMOTE(k_neighbors=1)), ("scaler", StandardScaler()), ("clf", clf)]

    elif model_name == "MLP":
        alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
        learning_rate_init = trial.suggest_loguniform("lr", 1e-4, 1e-1)
        clf = MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", solver="adam",
            alpha=alpha, learning_rate_init=learning_rate_init,
            early_stopping=True, max_iter=500, random_state=42
        )
        steps = [("smote", SMOTE(k_neighbors=1)), ("scaler", StandardScaler()), ("clf", clf)]

    elif model_name == "XGBoost":
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        clf = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
        steps = [("smote", SMOTE(k_neighbors=1)), ("clf", clf)]

    elif model_name == "SMOTEBoost":
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME", random_state=42
        )
        steps = [("smote", SMOTE(random_state=42, k_neighbors=1)), ("clf", clf)]

    elif model_name == "RUSBoost":
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME", random_state=42
        )
        steps = [("rus", RandomUnderSampler(random_state=42)), ("clf", clf)]

    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return Pipeline(steps=steps)
