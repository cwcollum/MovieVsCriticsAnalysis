import duckdb
import pandas as pd
import numpy as np

def load_384(
        embedding_dat_file: str,
        csv_file: str,
        table_name: str):
    """
    Load the embeddings from a .dat file, append them to their respective rows
    form a csv file, and save the results in a DuckDB table. Can't load the
    dataframe entirely into memory because it's too large.
    """
    print("Creating DuckDB table")
    cursor = duckdb.connect(f"{table_name}.db")
    cursor.execute(f"CREATE TABLE Embeddings (EmbeddingID INTEGER PRIMARY KEY, EmbeddingValue ARRAY<float>)")
    cursor.execute(f"CREATE TABLE {table_name} (Viewer VARCHAR, Movie VARCHAR, Sentiment VARCHAR, Reviews VARCHAR, EmbeddingID INTEGER, FOREIGN KEY (EmbeddingID) REFERENCES Embeddings(EmbeddingID))")
    print("Table created")
    embeddings = np.fromfile(embedding_dat_file, dtype=np.float32)
    while embeddings is not None:
        print("Loading embeddings")
        embeddings = embeddings.reshape(-1, 384)
        print("Embeddings loaded")
        print("Loading dataframe")
        data = pd.read_csv(csv_file)
        print("Dataframe loaded")
        print("Adding embeddings to dataframe")
        data['Embeddings'] = [embeddings[i] for i in range(len(data))]
        print("Embeddings added")
        print("Inserting data into DuckDB table")
        cursor.execute(f"INSERT INTO {table_name} VALUES {data}")
        print("Data inserted")
        embeddings = np.fromfile(embedding_dat_file, dtype=np.float32)
    return