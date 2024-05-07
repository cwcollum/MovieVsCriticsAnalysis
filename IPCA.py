# Script to take output384.csv and compute IPCA on the embeddings

from calendar import c
from chunk import Chunk
import pickle
import readline
import numpy as np
from torch import embedding
import tqdm
import csv
import os
import pickle
import duckdb

from sklearn.decomposition import IncrementalPCA as IPCA

pca = IPCA(n_components=100)
chunk_size = 4096

print("Connecting to DuckDB database")
cursor = duckdb.connect('embeddings.db')
print("Connected to DuckDB database")
print("Reading embeddings")
cursor.execute("SELECT EmbeddingValue, ID FROM embeddings384")
print("Convert embeddings to dataframe")
embeddings = cursor.fetchdf()
print("Embeddings converted to dataframe")
if os.path.exists('pca.pkl'):
    print("Loading PCA model")
    with open('pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    print("PCA model loaded")
else:
    print("Fitting PCA")
    for i in range(0, len(embeddings), chunk_size):
        print(f"Processing chunk {i // chunk_size} of {len(embeddings) // chunk_size}")
        chunk = embeddings[i:i + chunk_size]
        # Turn each row from a list of a single numpy array to just the numpy array
        chunk = chunk.to_numpy()
        chunk = np.array([row[0] for row in chunk])
        # Stack the numpy arrays into a single numpy array
        chunk = np.stack(chunk)
        pca.partial_fit(chunk)
        print(f"Chunk {i // chunk_size} processed")
    print("PCA fitted")
    # Save the PCA model
    print("Saving PCA model")
    with open('pca.pkl', 'wb') as file:
        pickle.dump(pca, file)
    print("PCA model saved")


print("Transforming embeddings")
# Add a new column to the dataframe to store the transformed embeddings
for i in range(0, len(embeddings), chunk_size):
    print(f"Transforming chunk {i // chunk_size} of {len(embeddings) // chunk_size}")
    max_end_guard = min(i + chunk_size, len(embeddings))
    chunk = embeddings[i:max_end_guard]
    chunk = chunk.to_numpy()
    chunk = np.array([row[0] for row in chunk])
    chunk = np.stack(chunk)
    chunk = pca.transform(chunk)
    # Break the chunk into rows
    for j in range(len(chunk)):
        embeddings.at[i + j, 'EmbeddingValue'] = chunk[j]
    print(f"Chunk {i // chunk_size} Transformed")

# Convert the numpy arrays to lists of floats
embeddings['EmbeddingValue'] = embeddings['EmbeddingValue'].apply(lambda x: x.tolist())
#
# Reinsert the embeddings into the table
# Use the ID column to match the embeddings
print("Inserting PCA embeddings into DuckDB table")
# Add the dataframes to the table
try:
    cursor.execute("ALTER TABLE embeddings384 ADD COLUMN EmbeddingValue100 FLOAT[100]")
except:
    pass
cursor.execute("UPDATE embeddings384 SET EmbeddingValue100 = embeddings.EmbeddingValue FROM embeddings WHERE embeddings384.ID = embeddings.ID")
print("PCA embeddings inserted into DuckDB table")
