import pandas as pd
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Optional, for balancing classes
from tqdm import tqdm
import warnings
import os

def check_class_imbalance(y, label="Dataset"):
    print(f"\n==== Class Frequency in {label} ====")
    print(y.value_counts())
    print(f"\n==== Class Proportion in {label} (%) ====")
    print((y.value_counts(normalize=True) * 100).round(2))

def main():
    warnings.filterwarnings("ignore")

    # ========== CONFIG ==========
    DATA_PATH = 'E:/HPC-ML/q/HIGGS.csv'
    SUBSET_SIZE = 100_000     # Adjust as needed
    TARGET_COL = 0
    USE_SMOTE = False         # Set to True if you want to balance with SMOTE
    RANDOM_STATE = 42
    # ============================

    print("📥 Loading data subset...")
    df = pd.read_csv(DATA_PATH, header=None, nrows=SUBSET_SIZE)
    print(f"🗒️ Loaded shape: {df.shape}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Check imbalance in full subset
    check_class_imbalance(y, "Full Subset")

    non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Found non-numeric columns in features: {non_numeric}")

    print("🔀 Splitting train/val (stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Check imbalance in train and validation sets
    check_class_imbalance(pd.Series(y_train), "Train Set")
    check_class_imbalance(pd.Series(y_val), "Validation Set")

    print("📊 Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    if USE_SMOTE:
        print("⚖️ Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # Check imbalance after SMOTE
        check_class_imbalance(pd.Series(y_train), "Train Set After SMOTE")

    max_trees = 50
    patience = 5
    best_acc = 0.0
    no_improve = 0
    best_model = None
    print("⚙️ Training RandomForestClassifier (with early stopping)...")

    for n in tqdm(range(1, max_trees + 1), desc="n_estimators"):
        model = RandomForestClassifier(
            n_estimators=n,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            best_model = pickle.dumps(model)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"⏹️ Early stopping at {n} trees. Best val accuracy: {best_acc:.4f}")
            break

    final_model = pickle.loads(best_model)
    out_path = "model_subset.pkl"
    joblib.dump(final_model, out_path)
    print(f"✅ Model saved to {out_path} | Best accuracy: {best_acc:.4f}")
    print(classification_report(y_val, final_model.predict(X_val), digits=3))

if __name__ == "__main__":
    main()
