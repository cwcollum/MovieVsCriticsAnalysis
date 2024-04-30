# Script to take output384.csv and compute IPCA on the embeddings

import readline
import numpy as np
import tqdm
import csv
import os

from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=100)
chunk_size = 250
count = 0
if os.path.exists('output384_size.num'):
    with open('output384_size.num', 'r') as f:
        count = int(f.read())
else:
    with open('output384.csv', 'r') as f:
        count = sum(1 for line in f)
    with open('output384_size.num', 'w') as f:
        f.write(str(count))
count -= 1 # Subtract 1 for the header
with open('output384.csv', 'r') as f:
    # Skip headers
    next(f)
    with tqdm.tqdm(total=count // chunk_size) as progress_bar:
        for _ in range(count // chunk_size):
            # Gather a chunk of valid lines
            chunk = []
            while len(chunk) < chunk_size:
                line = f.readline()
                if line == '':
                    break
                line = line.split(',')[4:]
                # Remove rows with missing values
                if line[0] == '':
                    continue
                line = [float(x) for x in line]
                chunk.append(line)
            if len(chunk) < chunk_size:
                break
            chunk = np.array(chunk, dtype=float)
            ipca.partial_fit(chunk)
            progress_bar.update(1)

# Go back through the data and transform itwith open('output384.csv', 'r') as f:
with open('output384.csv', 'r') as f:
    with open('output100.csv', 'w') as f_out:
        f_out.write('Viewer,Movie,Sentiment,Review,Embedding\n')
        with open('output384.csv', 'r') as f:
            # Skip headers
            next(f)
            with tqdm.tqdm(total=count // chunk_size + 1) as progress_bar:
                while True:
                    # Gather a chunk of valid lines
                    chunk = []
                    other_vals = []
                    while len(chunk) < chunk_size:
                        line = f.readline()
                        if line == '':
                            break
                        other_val = line.split(',')[:4]
                        line = line.split(',')[4:]
                        # Remove rows with missing values
                        if line[0] == '':
                            continue
                        other_vals.append(other_val)
                        line = [float(x) for x in line]
                        chunk.append(line)
                    chunk = np.array(chunk, dtype=float)
                    chunk = ipca.transform(chunk)
                    for i, line in enumerate(chunk):
                        f_out.write(','.join(other_vals[i]) + ',')
                        f_out.write(' '.join([str(x) for x in line]) + '\n')
                    progress_bar.update(1)
                    if len(chunk) < chunk_size:
                        break
                