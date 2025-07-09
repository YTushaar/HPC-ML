import pandas as pd
import joblib
import time

def main():
    # 1) Load
    df = pd.read_csv('data.csv')

    # 2) Drop any non-feature columns
    drop_cols = []
    for col in ['Unnamed: 0', 'id', 'target']:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols)

    # 3) Load model
    model = joblib.load('model.pkl')

    # 4) Predict & time
    start = time.time()
    preds = model.predict(X)
    duration = time.time() - start

    print(f"Serial inference took {duration:.2f} s")

    # 5) Save
    pd.DataFrame({'prediction': preds}) \
      .to_csv('predictions_serial.csv', index=False)

if __name__ == "__main__":
    main()
