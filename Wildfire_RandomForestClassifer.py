# Riley Albee
# 10/20/2024
# Machine Learning: CS379
# Description:
#

"""
This script performs fire classification using Random Forest on wildfire data.

It processes data from a SQLite database, trains a Random Forest model,
and generates visualizations of the results.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_chunk(chunk):
    """
    Preprocess a chunk of data from the wildfire dataset.

    Args:
    chunk (pd.DataFrame): A chunk of the wildfire dataset

    Returns:
    tuple: Preprocessed feature matrix (X) and target variable (y)
    """
    chunk['DISCOVERY_DATE'] = pd.to_datetime(chunk['DISCOVERY_DATE'])
    chunk['MONTH'] = chunk['DISCOVERY_DATE'].dt.month
    chunk['TARGET'] = chunk['STAT_CAUSE_DESCR'].apply(
        lambda x: 0 if x == 'Lightning' else 1
    )

    features = [
        'FIRE_YEAR', 'DISCOVERY_DOY', 'DISCOVERY_TIME', 'FIRE_SIZE',
        'LATITUDE', 'LONGITUDE', 'MONTH'
    ]

    categorical_features = ['FIRE_SIZE_CLASS', 'OWNER_CODE', 'STATE', 'COUNTY']
    for feature in categorical_features:
        le = LabelEncoder()
        chunk[feature] = le.fit_transform(chunk[feature].astype(str))
        features.append(feature)

    X = chunk[features]
    y = chunk['TARGET']

    X = X.dropna(axis=1, how='all')

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y


def plot_feature_importance(importance, names, model_type):
    """Create a bar plot of feature importances."""
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'{model_type} FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """Create a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    """Plot the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(y_true, y_pred_proba):
    """Plot the distribution of predicted probabilities for each class."""
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba[y_true == 0], kde=True, color="skyblue", label="Natural", bins=50)
    sns.histplot(y_pred_proba[y_true == 1], kde=True, color="red", label="Man-made", bins=50)
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability of Man-made Fire")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the fire classification script."""
    conn = sqlite3.connect('FPA_FOD_20170508.sqlite')

    X_list = []
    y_list = []

    query = """
    SELECT FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME,
           FIRE_SIZE, FIRE_SIZE_CLASS, LATITUDE, LONGITUDE, OWNER_CODE,
           STATE, COUNTY, STAT_CAUSE_DESCR
    FROM Fires
    """

    total_rows = pd.read_sql_query("SELECT COUNT(*) FROM Fires", conn).iloc[0, 0]

    chunk_size = 10000
    with tqdm(total=total_rows, desc="Processing chunks") as pbar:
        for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
            X_chunk, y_chunk = preprocess_chunk(chunk)
            X_list.append(X_chunk)
            y_list.append(y_chunk)
            pbar.update(len(chunk))

    X = pd.concat(X_list)
    y = pd.concat(y_list)

    conn.close()

    print("\nFinal features:", X.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining Random Forest Classifier...")
    n_estimators = 100
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    fit_original = rf_classifier.fit

    def fit_with_progress(*args, **kwargs):
        with tqdm(total=n_estimators, desc="Training trees") as pbar:
            result = fit_original(*args, **kwargs)
            for _ in range(n_estimators):
                pbar.update(1)
        return result

    rf_classifier.fit = fit_with_progress

    rf_classifier.fit(X_train, y_train)

    print("Making predictions...")
    y_pred = rf_classifier.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Natural', 'Man-made']))

    print("\nGenerating visualizations...")

    plot_feature_importance(rf_classifier.feature_importances_, X.columns, 'RANDOM FOREST')
    plot_confusion_matrix(y_test, y_pred, classes=['Natural', 'Man-made'])

    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_pred_proba)
    plot_probability_distribution(y_test, y_pred_proba)

    plt.close('all')
    figures = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figures):
        fig.savefig(f'figure_{i+1}.png')
        print(f"Figure {i+1} saved as figure_{i+1}.png")

    print("\nAll visualizations have been generated and saved.")


if __name__ == "__main__":
    main()