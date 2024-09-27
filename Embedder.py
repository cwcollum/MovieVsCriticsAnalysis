# %%


# %%
import csv
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

# %%
def create_dataframe():
    classes = [
        'a_hate_c_hate',
        'a_hate_c_love',
        'a_love_c_hate',
        'a_love_c_love']
    # Assuming movies are in movie_list.csv
    print("Reading movie list")
    movies = []
    with open('movie_list.csv', 'r') as file:
        for line in file:
            movies.append(line.strip())

    # Create one pandas dataframe for all the data
    # Viewer, Movie, Positive/Negative, Review
    p_bar = tqdm(total=len(movies) * 2)
    data = pd.DataFrame(columns=['Viewer', 'Movie', 'Sentiment', 'Reviews'])
    for i in range(len(classes)):
        for j in range(len(movies)):
            filename_a = f'data/{classes[i]}/{movies[j]}/a_{movies[j]}.csv'
            filename_c = f'data/{classes[i]}/{movies[j]}/c_{movies[j]}.csv'
            # Read the data. Movies are only ever in one class
            print(f"Reading {filename_a} and {filename_c}")
            print(f"Percentage of files read: {100 * (i * len(movies) + j) / (len(classes) * len(movies))}%")
            try:
                data_a = pd.read_csv(filename_a, header=0)
                data_a['Viewer'] = 'Audience'
                data_a['Movie'] = movies[j]
                if classes[i] == 'a_hate_c_hate' or classes[i] == 'a_hate_c_love':
                    data_a['Sentiment'] = 'Negative'
                else:
                    data_a['Sentiment'] = 'Positive'
                data_a = data_a[['Viewer', 'Movie', 'Sentiment', 'Reviews']]
                data = pd.concat([data, data_a])
                p_bar.update(1)
            except FileNotFoundError:
                pass
            # End of try except block
            try:
                data_c = pd.read_csv(filename_c, header=0)
                data_c['Viewer'] = 'Critic'
                data_c['Movie'] = movies[j]
                if classes[i] == 'a_hate_c_hate' or classes[i] == 'a_love_c_hate':
                    data_c['Sentiment'] = 'Negative'
                else:
                    data_c['Sentiment'] = 'Positive'
                data_c = data_c[['Viewer', 'Movie', 'Sentiment', 'Reviews']]
                data = pd.concat([data, data_c])
                p_bar.update(1)
            except FileNotFoundError:
                pass
            # End of try except block
        # End of for j in range(len(movies))
    # End of for i in range(len(classes))
    p_bar.close()

    # Save the combined data to a CSV file
    data.to_csv('data_combined.csv', index=False)

# %%
# Make folder data_tokens if it does not exist
if not os.path.exists('data_tokens'):
    os.makedirs('data_tokens')

# %%

if not os.path.exists('data_combined.csv'):
    create_dataframe()

# %%
MODEL_EXISTS = True
if not os.path.exists('word2vec.model'):
    MODEL_EXISTS = False

# %%

# Load the data
data = pd.read_csv('data_combined.csv')

# %%


# %%
if not MODEL_EXISTS:
    with open('data_combined.csv', 'r') as file:
        data = pd.read_csv(file)
        # reviews is a list of strings
        reviews = data['Reviews'].tolist()
        # Split each review into words
        for i in range(len(reviews)):
            reviews[i] = reviews[i].split()
        # Embed the data using GenSim Word2Vec
        cores = multiprocessing.cpu_count()
        model = Word2Vec(min_count=20,
                        vector_size=100,
                        window=2,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)
        model.build_vocab(reviews, progress_per=len(reviews))
        model.train(reviews, total_examples=model.corpus_count, epochs=30, report_delay=1)
        model.save('word2vec.model')
# End of if not MODEL_EXISTS

# %%
# Load the model
model = Word2Vec.load('word2vec.model')

# %%
# Convert the reviews to embeddings by averaging the word embeddings
# Work on dataframe data

headers = ['Viewer', 'Movie', 'Sentiment']
with open('data_combined.csv', 'r') as input_file:
    with open('data_embeddings.csv', 'w') as output_file:
        # Have tqdm to show progress
        reader = csv.reader(input_file)
        next(reader)
        writer = csv.writer(output_file)
        writer.writerow(headers + [f'embedding_{i}' for i in range(100)])
        with tqdm(total=len(data)) as p_bar:
            for row in tqdm(reader):
                review = row[3].split()
                embedding = [model.wv[word] for word in review if word in model.wv]
                if len(embedding) == 0:
                    embedding = np.zeros(shape=(1, 100))
                embedding = np.array(embedding)
                embedding = embedding.mean(axis=0)
                writer.writerow(row[0:3] + list(embedding))
                p_bar.update(1)
            # End of for row in tqdm(reader)
        # End of with tqdm(total=len(data)) as p_bar:
    # End of with open('data_embeddings.csv', 'w') as output_file:
# End of with open('data_combined.csv', 'r') as input_file: