import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # 1) Load data without header (HIGGS dataset has no column names)
    df = pd.read_csv('E:/HPC-ML/q/HIGGS.csv', header=None)
    print(f"🗒️ Loaded shape: {df.shape}")

    # 2) Define target and features
    target_col = 0  # First column is the target (0 or 1)
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # 3) Ensure all features are numeric
    non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Found non-numeric columns in features: {non_numeric}")

    # 4) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Train classifier
    print("⚙️  Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    # 6) Save model
    joblib.dump(model, 'model.pkl')
    print("✅ Model trained and saved to model.pkl")

if __name__ == "__main__":
    main()
