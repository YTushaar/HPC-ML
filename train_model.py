import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    # 1) Load data & inspect columns
    df = pd.read_csv('data.csv')
    print("🗒️ Columns in CSV:", df.columns.tolist())

    # 2) Pull out target, drop id + target from df
    target_col = 'target'
    if 'id' in df.columns:
        df_features = df.drop(columns=['id', target_col])
    else:
        df_features = df.drop(columns=[target_col])

    # 3) Extract y separately
    y = df[target_col]

    # 4) Ensure X is all numeric
    non_numeric = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Found non-numeric columns in features: {non_numeric}")

    X = df_features

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) Choose regressor vs classifier based on y dtype
    if pd.api.types.is_float_dtype(y):
        print("⚙️  Detected continuous target → using RandomForestRegressor")
        model = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42
        )
    else:
        print("⚙️  Detected discrete target → using RandomForestClassifier")
        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42
        )

    # 7) Fit and save
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    print("✅ Model trained and saved to model.pkl")

if __name__ == "__main__":
    main()
