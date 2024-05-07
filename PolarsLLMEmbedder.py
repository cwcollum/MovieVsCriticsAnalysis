# %%
import polars as pl
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import tqdm
import os
import numpy as np
from npy_append_array import NpyAppendArray
from sklearn.decomposition import PCA
from utils.meanpooling import mean_pooling

# %%
model_id = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 200

# %%
# Load model from HuggingFace Hub
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
# Compile model
model = torch.compile(model)
# Set model to evaluation mode
model.eval()
# Move model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Model loaded.")

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

if os.path.exists('data_minilm_tokenized.csv'):
    print("Tokens already exist.")
else:
    # %%
    print("Loading data...")
    df = pl.read_csv('data_combined.csv')
    print("Getting sentences...")
    sentences: list[str] = df['Reviews'].to_list()

    # %%
    # Tokenize sentences
    print("Tokenizing sentences...")
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors='np')

    # Save tokenized sentences to new csv with headers:
    # Viewer,Movie,Sentiment,Review,Tokenized
    print("Saving tokenized sentences...")
    # Convert Tokenized and AttentionMask to strings
    df = df.with_columns(
        Tokenized = [np.array_str(x, max_line_width=np.inf) for x in encoded_input['input_ids']],
        AttentionMask = [np.array_str(x, max_line_width=np.inf) for x in encoded_input['attention_mask']])
    del encoded_input

    df.write_csv('data_minilm_tokenized.csv')
    del df
# End else
exit()
print("Setting up PyTorch...")
torch.set_float32_matmul_precision('high')
# %%
count = 0
with open('data_minilm_tokenized.csv', 'r') as f:
    count = sum(1 for _ in f)
# Compute token embeddings
# Output is too large to fit in memory, so we dump it to a file
print("Computing embeddings...")
with open('output384.csv', 'w') as f:
    # Add headers
    f.write('Viewer,Movie,Sentiment,Review,Embedding\n')
    with torch.no_grad():
        # Stream batches from csv. wrapped in tqdm for progress bar.
        with tqdm.tqdm(total=count // BATCH_SIZE) as progress_bar:
            reader = pl.read_csv_batched('data_minilm_tokenized.csv')
            batch = reader.next_batches(BATCH_SIZE)
            while batch:
                df = pl.concat(batch)
                attention_mask = df['AttentionMask']
                tokens = df['Tokenized']
                attention_mask = [x.strip('][').split(', ') for x in attention_mask]
                tokens = [x.strip('][').split(', ') for x in tokens]
                attention_mask = [np.array(x, dtype=bool) for x in attention_mask]
                tokens = [np.array(x, dtype=int) for x in tokens]
                attention_mask = np.array(attention_mask)
                tokens = np.array(tokens)
                attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
                tokens = torch.tensor(tokens, dtype=torch.int)
                # Move batch to GPU, if available
                attention_mask = attention_mask.to(device)
                tokens = tokens.to(device)
                # Forward pass
                output = model(tokens)
                
                output = mean_pooling(output, attention_mask)
                output = F.normalize(output, p=2, dim=1)
                # Move output to CPU
                output = output.to('cpu').detach().clone().numpy()
                # Avoid NaNs
                output = np.nan_to_num(output)
                # Make mini-dataframe
                df = df.with_columns(
                    Embedding = [np.array_str(x, max_line_width=np.inf) for x in output])
                # Save to file
                df.write_csv(f, include_header=False)
                # Free up memory
                del output
                del df
                del attention_mask
                torch.cuda.empty_cache()
                # Increment tqdm progress bar
                progress_bar.update(1)
                # Get next batch
                batch = reader.next_batches(BATCH_SIZE)
            # End for i, batch in enumerate(pd.read_csv('data_minilm_tokenized.csv', chunksize=BATCH_SIZE))
        # End with tqdm(total=len_of_my_iterable) as progress_bar
    # End with torch.no_grad()
# End with open('output384.csv', 'w') as f
print("Embeddings computed.")