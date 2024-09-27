# %%
import pandas as pd
from polars import first
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn.functional as F
import tqdm
import os
import numpy as np
from sklearn.decomposition import PCA
import asyncio
import duckdb

# %%
model_id = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512

# %%
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_model(model_id: str) -> tuple[
    torch.nn.Module,
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    torch.device]:
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
    return model, tokenizer, device # type: ignore

def generate_tokens(
        in_file: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        chunk_size: int) -> None:
    if not os.path.exists('data_tokens'):
        os.makedirs('data_tokens')

    # Check if any files are in the directory
    if os.listdir('data_tokens'):
        print("Tokens already exist.")
    else:
        print("Getting sentences...")
        with open(in_file, 'r') as f:
            count = sum(1 for _ in f) - 1
        num_of_chunks = count // chunk_size
        if count % chunk_size != 0:
            num_of_chunks += 1
        with open(in_file, 'r') as f:
            f.readline() # Skip header
            for i in range(num_of_chunks):
                # Read chunk
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(f.readline())
                    except StopIteration:
                        break
                # Tokenize chunk
                encoded_input = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors='np')
                # Save Tokenized and AttentionMask to npy files
                first_index = i * chunk_size
                last_index = first_index + len(encoded_input['input_ids'])
                with open(f'data_tokens/{first_index}_to_{last_index}_tokens.npy', 'wb') as g:
                    np.save(g, encoded_input['input_ids'])
                with open(f'data_tokens/{first_index}_to_{last_index}_attention_mask.npy', 'wb') as g:
                    np.save(g, encoded_input['attention_mask'])
            # End for i in range(num_of_chunks)
        # End with open(in_file, 'r') as f
    # End else
# End generate_tokens

model, tokenizer, device = load_model(model_id)
generate_tokens('data_combined.csv', tokenizer, BATCH_SIZE)

print("Setting up PyTorch...")
torch.set_float32_matmul_precision('high')

# Can extract number of samples from the last chunked file
count = 0
for file in os.listdir('data_tokens'):
    if int(file.split('_')[2]) > count:
        count = int(file.split('_')[2])
# Compute token embeddings
# Output is too large to fit in memory, so we dump it to a DuckDB
print("Computing embeddings...")
# Create DuckDB table
cursor = duckdb.connect('embeddings.db')
cursor.execute(f"CREATE TABLE embeddings384 (Viewer VARCHAR, Movie VARCHAR, Sentiment VARCHAR, Reviews VARCHAR, EmbeddingValue FLOAT[384])")
cursor.execute("CREATE SEQUENCE seq_id START 1;")
cursor.execute("ALTER TABLE embeddings384 ADD COLUMN ID INTEGER DEFAULT nextval('seq_id');")
with torch.no_grad():
    # Stream batches from csv. wrapped in tqdm for progress bar.
    with tqdm.tqdm(total=count // BATCH_SIZE + 1) as progress_bar:
        for i in range(0, count, BATCH_SIZE):
            # Read tokens and attention masks in batch
            tokens = []
            attention_mask = []
            try:
                tokens = np.load(f'data_tokens/{i}_to_{i + BATCH_SIZE}_tokens.npy')
                attention_mask = np.load(f'data_tokens/{i}_to_{i + BATCH_SIZE}_attention_mask.npy')
            except FileNotFoundError:
                print(f"File {i}_to_{i + BATCH_SIZE} not found.")
                continue
            # End try
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
            tokens = torch.tensor(tokens, dtype=torch.int)
            # Move batch to GPU, if available
            attention_mask = attention_mask.to(device)
            tokens = tokens.to(device)
            # Forward pass
            output = model(tokens)
            del tokens
            
            output = mean_pooling(output, attention_mask)
            del attention_mask
            output = F.normalize(output, p=2, dim=1)
            # Move output to CPU
            output = output.to('cpu').detach().clone().numpy()
            # Avoid NaNs
            output = np.nan_to_num(output)
            # Create subtable in pandas
            data = pd.read_csv('data_combined.csv', skiprows=i, nrows=BATCH_SIZE, header=0)
            data['EmbeddingValue'] = [output[j] for j in range(len(data))]
            # Insert subtable into DuckDB
            cursor.execute(f"INSERT INTO embeddings384 SELECT * FROM data")
            # Free up memory
            del output
            torch.cuda.empty_cache()
            # Increment tqdm progress bar
            progress_bar.update(1)
        # End for i, batch in enumerate(pd.read_csv('data_minilm_tokenized.csv', chunksize=BATCH_SIZE))
    # End with tqdm(total=len_of_my_iterable) as progress_bar
# End with torch.no_grad()
print("Embeddings computed.")