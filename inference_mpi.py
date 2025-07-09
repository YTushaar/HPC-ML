from mpi4py import MPI
import pandas as pd
import joblib
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 reads & prepares X
    if rank == 0:
        df = pd.read_csv('data.csv')

        # drop any non-feature columns
        drop_cols = [c for c in ['Unnamed: 0', 'id', 'target'] if c in df.columns]
        X = df.drop(columns=drop_cols)

        # split into 'size' chunks
        splits = np.array_split(X, size)
    else:
        splits = None

    # scatter the DataFrame chunks
    X_chunk = comm.scatter(splits, root=0)

    # load model once per process
    model = joblib.load('model.pkl')

    # local timing & predict
    start = MPI.Wtime()
    preds_chunk = model.predict(X_chunk)
    local_time = MPI.Wtime() - start
    print(f"[Rank {rank}] inference: {local_time:.2f} s")

    # gather predictions and timings
    all_preds = comm.gather(preds_chunk, root=0)
    all_times = comm.gather(local_time, root=0)

    if rank == 0:
        # concatenate and save
        flat = np.concatenate(all_preds)
        pd.DataFrame({'prediction': flat}) \
          .to_csv('predictions_mpi.csv', index=False)
        print(f"📊 MPI inference on {size} procs, max time: {max(all_times):.2f} s")

if __name__ == "__main__":
    main()
