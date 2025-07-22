import os
import warnings
import pickle
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    adjusted_rand_score,
    homogeneity_score,
)
from imblearn.over_sampling import SMOTE

def check_class_imbalance(y, label="Dataset"):
    print(f"\n==== Class Frequency in {label} ====")
    print(y.value_counts())
    print(f"\n==== Class Proportion in {label} (%) ====")
    print((y.value_counts(normalize=True) * 100).round(2))

def main():
    warnings.filterwarnings("ignore")

    # ========== CONFIG ==========
    DATA_PATH    = 'D:/HPC-ML/HIGGS.csv'
    SUBSET_SIZE  = 100_000       # adjust as needed
    TARGET_COL   = 0
    USE_SMOTE    = False         # balance train set?
    RANDOM_STATE = 42
    # ============================

    print("📥 Loading data subset...")
    df = pd.read_csv(DATA_PATH, header=None, nrows=SUBSET_SIZE)
    print(f"🗒️ Loaded shape: {df.shape}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    check_class_imbalance(y, "Full Subset")

    # all features must be numeric
    non_numeric = X.select_dtypes(include=['object','category']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Found non-numeric columns: {non_numeric}")

    print("🔀 Splitting train/val (stratified)…")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    check_class_imbalance(y_train, "Train Set")
    check_class_imbalance(y_val,   "Validation Set")

    print("📊 Scaling features…")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    if USE_SMOTE:
        print("⚖️  Applying SMOTE…")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)
        check_class_imbalance(y_train, "Train After SMOTE")

    # === Supervised classifiers ===
    classifiers = {
        "Naive Bayes"       : GaussianNB(),
        "Decision Tree"     : DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(random_state=RANDOM_STATE),
        "AdaBoost"          : AdaBoostClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting" : GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Bagging": BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=RANDOM_STATE
),
        "Random Forest"     : RandomForestClassifier(random_state=RANDOM_STATE),
    }

    print("\n⚙️  Training & evaluating classifiers:")
    results = {}
    for name, clf in classifiers.items():
        print(f"\n>> {name}")
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_val_sc)
        acc    = accuracy_score(y_val, y_pred)
        print(f"  • Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred, digits=3))
        results[name] = acc
        # optionally save each model:
        with open(f"{name.replace(' ','_')}.pkl", "wb") as f:
            pickle.dump(clf, f)

    # === Unsupervised clusterers ===
    n_clusters = len(np.unique(y_train))
    clusterers = {
        "KMeans"           : KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE),
        "SpectralClustering": SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_STATE),
        "AffinityPropagation": AffinityPropagation(),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE),
    }

    print("\n⚙️  Fitting & evaluating clusterers (on VAL set):")
    for name, clust in clusterers.items():
        print(f"\n>> {name}")
        # fit & predict cluster labels on validation features
        labs = clust.fit_predict(X_val_sc)
        ari  = adjusted_rand_score(y_val, labs)
        homo = homogeneity_score(y_val, labs)
        print(f"  • Adjusted Rand Index: {ari:.4f}")
        print(f"  • Homogeneity Score:   {homo:.4f}")

    print("\n✅ All models trained and evaluated.")

if __name__ == "__main__":
    main()
