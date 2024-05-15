import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import duckdb
import seaborn as sns

# Connect to the database
db = duckdb.connect('embeddings.db')
# Load the data
df = db.execute("SELECT Sentiment, EmbeddingValue FROM embeddings384 WHERE Viewer = 'Audience'").fetchdf()
df100 = db.execute("SELECT Sentiment, EmbeddingValue100 FROM embeddings384 WHERE Viewer = 'Audience'").fetchdf()

tsne = TSNE(n_components=2, verbose=1)
reduced_data = tsne.fit_transform(
    np.array(df100['EmbeddingValue100'].tolist()),
    df100['Sentiment'].tolist())
print(len(reduced_data))
print(reduced_data.shape)
# convert to dataframe
reduced_data = pd.DataFrame(
    data = {
        'x': reduced_data[:,0],
        'y': reduced_data[:,1],
        'sentiment': df100['Sentiment']
    }
)
p = sns.jointplot(
    data=reduced_data, x="x", y="y", hue='sentiment', palette=['red', 'green']
)
p.figure.suptitle('t-SNE of Audience Embeddings (100 -> 2)')
p.ax_joint.collections[0].set_alpha(1)
p.figure.tight_layout()
# Increase font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
p.figure.subplots_adjust(top=0.95) # Reduce plot to make room 
# Adjust x and y labels
p.set_axis_labels('t-SNE 1', 't-SNE 2', fontsize=12)
plt.savefig('t-SNE-Audience-100-2.png')
p = sns.jointplot(
    data=reduced_data, x="x", y="y", hue='sentiment', palette=['green', 'red']
)
p.figure.suptitle('t-SNE of Audience Embeddings (100 -> 2)')
p.ax_joint.collections[0].set_alpha(1)
p.figure.tight_layout()
# Increase font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
p.figure.subplots_adjust(top=0.95) # Reduce plot to make room 
# Adjust x and y labels
p.set_axis_labels('t-SNE 1', 't-SNE 2', fontsize=12)
plt.savefig('t-SNE-Audience-100-2-alt.png')


# Load the data
df = db.execute("SELECT Sentiment, EmbeddingValue FROM embeddings384 WHERE Viewer = 'Critic'").fetchdf()
df100 = db.execute("SELECT Sentiment, EmbeddingValue100 FROM embeddings384 WHERE Viewer = 'Critic'").fetchdf()

tsne = TSNE(n_components=2, verbose=1)
reduced_data = tsne.fit_transform(
    np.array(df100['EmbeddingValue100'].tolist()),
    df100['Sentiment'].tolist())
print(len(reduced_data))
print(reduced_data.shape)
# convert to dataframe
reduced_data = pd.DataFrame(
    data = {
        'x': reduced_data[:,0],
        'y': reduced_data[:,1],
        'sentiment': df100['Sentiment']
    }
)
colors = ['green' if x == 'Positive' else 'red' for x in reduced_data['sentiment']]
p = sns.jointplot(
    data=reduced_data, x="x", y="y", hue='sentiment', palette=['red', 'green']
)
p.figure.suptitle('t-SNE of Critic Embeddings (100 -> 2)')
p.ax_joint.collections[0].set_alpha(1)
p.figure.tight_layout()
# Increase font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
p.figure.subplots_adjust(top=0.95) # Reduce plot to make room 
# Adjust x and y labels
p.set_axis_labels('t-SNE 1', 't-SNE 2', fontsize=12)
plt.savefig('t-SNE-Critic-100-2.png')

p = sns.jointplot(
    data=reduced_data, x="x", y="y", hue='sentiment', palette=['green', 'red']
)
p.figure.suptitle('t-SNE of Critic Embeddings (100 -> 2)')
p.ax_joint.collections[0].set_alpha(1)
p.figure.tight_layout()
# Increase font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
p.figure.subplots_adjust(top=0.95) # Reduce plot to make room 
# Adjust x and y labels
p.set_axis_labels('t-SNE 1', 't-SNE 2', fontsize=12)
plt.savefig('t-SNE-Critic-100-2-alt.png')


