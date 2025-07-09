import pandas as pd
import joblib
from joblib import Parallel, delayed
import multiprocessing
import time

def chunk_predict(model, X_chunk):
    return model.predict(X_chunk)

def main():
    # 1) Load raw data
    df = pd.read_csv('data.csv')

    # 2) Drop any non-feature cols: Unnamed: 0, id, target
    drop_cols = [c for c in ['Unnamed: 0', 'id', 'target'] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # 3) Load your trained model
    model = joblib.load('model.pkl')

    # 4) Split into roughly equal chunks
    n_cores = multiprocessing.cpu_count()
    chunk_size = len(X) // n_cores + 1
    splits = [ X[i*chunk_size : (i+1)*chunk_size] for i in range(n_cores) ]

    # 5) Run parallel predict + time it
    start = time.time()
    results = Parallel(n_jobs=n_cores)(
        delayed(chunk_predict)(model, chunk) for chunk in splits
    )
    duration = time.time() - start

    # 6) Flatten and save
    total_preds = [pred for chunk in results for pred in chunk]
    pd.DataFrame({'prediction': total_preds}) \
      .to_csv('predictions_parallel.csv', index=False)

    print(f"Parallel inference with {n_cores} cores took {duration:.2f} s")

if __name__ == "__main__":
    main()
