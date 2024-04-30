# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import tqdm
import os
import numpy as np
from npy_append_array import NpyAppendArray
from sklearn.decomposition import PCA

# %%
model_id = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 200

# %%
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
    df = pd.read_csv('data_combined.csv')
    print("Getting sentences...")
    sentences = df['Reviews'].tolist()

    # %%
    # Tokenize sentences
    print("Tokenizing sentences...")
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Save tokenized sentences to new csv with headers:
    # Viewer,Movie,Sentiment,Review,Tokenized
    print("Saving tokenized sentences...")
    df['Tokenized'] = encoded_input['input_ids'].numpy().tolist()
    df['AttentionMask'] = encoded_input['attention_mask'].numpy().tolist()
    del encoded_input
    df.to_csv('data_minilm_tokenized.csv', index=False)
    del df
# End else
print("Setting up PyTorch...")
torch.set_float32_matmul_precision('high')
# %%
count = 0
with open('data_minilm_tokenized.csv', 'r') as f:
    count = sum(1 for line in f)
# Compute token embeddings
# Output is too large to fit in memory, so we dump it to a file
print("Computing embeddings...")
with open('output384.csv', 'w') as f:
    # Add headers
    f.write('Viewer,Movie,Sentiment,Review,Embedding\n')
    with torch.no_grad():
        # Stream batches from csv. wrapped in tqdm for progress bar.
        with tqdm.tqdm(total=count // BATCH_SIZE) as progress_bar:
            for i, batch in tqdm.tqdm(enumerate(pd.read_csv('data_minilm_tokenized.csv', chunksize=BATCH_SIZE))):
                # Convert Tokenized column to tensor
                other_vals = batch[['Viewer', 'Movie', 'Sentiment', 'Reviews']]
                attention_mask = batch['AttentionMask'].values
                batch = batch['Tokenized'].values
                attention_mask = [x.strip('][').split(', ') for x in attention_mask]
                batch = [x.strip('][').split(', ') for x in batch]
                attention_mask = [np.array(x, dtype=bool) for x in attention_mask]
                batch = [np.array(x, dtype=int) for x in batch]
                attention_mask = np.array(attention_mask)
                batch = np.array(batch)
                attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
                batch = torch.tensor(batch, dtype=torch.int)
                # Move batch to GPU, if available
                attention_mask = attention_mask.to(device)
                batch = batch.to(device)
                # Forward pass
                output = model(batch)
                
                output = mean_pooling(output, attention_mask)
                output = F.normalize(output, p=2, dim=1)
                # Move output to CPU
                output = output.to('cpu').detach().clone().numpy()
                # Avoid NaNs
                output = np.nan_to_num(output)
                # Make mini-dataframe
                output = pd.DataFrame(output)
                output = pd.concat([other_vals, output], axis=1)
                # Save to file
                output.to_csv(f, mode='a', header=False, index=False)
                # Free up memory
                del output
                del batch
                del attention_mask
                torch.cuda.empty_cache()
                # Increment tqdm progress bar
                progress_bar.update(1)
            # End for i, batch in enumerate(pd.read_csv('data_minilm_tokenized.csv', chunksize=BATCH_SIZE))
        # End with tqdm(total=len_of_my_iterable) as progress_bar
    # End with torch.no_grad()
# End with open('output384.csv', 'w') as f
print("Embeddings computed.")