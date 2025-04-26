from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_logistic_regression(X_train, y_train, random_state=42):
    """Trains a Logistic Regression model with balanced class weights."""
    model = LogisticRegression(solver='liblinear', random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100, class_weight='balanced'):
    """Trains a Random Forest Classifier model with balanced class weights."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1):
    """Trains a Gradient Boosting Classifier model."""
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1, class_weight='balanced'):
    """Trains a LightGBM Classifier model."""
    model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints key metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC: {auc_roc:.4f}")
    return y_pred, y_pred_proba