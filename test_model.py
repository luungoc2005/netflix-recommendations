import torch
from tqdm.auto import tqdm
import torch

import pandas as pd
from tqdm import tqdm

from collections import deque

from utils import DotDict

CANCEL_KEY = 'exit'

# Load a movie metadata dataset
movie_metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
# Remove the long tail of rarly rated moves
movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)

print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))

# Load data for all movies
movie_titles = pd.read_csv('./netflix-prize-data/movie_titles.csv', 
                           encoding = 'ISO-8859-1', 
                           header = None, 
                           names = ['Id', 'Year', 'Name']) #.set_index('Id')

print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))

df = pd.read_csv('user_ratings.csv')

print('Shape User-Ratings:\t{}'.format(df.shape))

# Create user- & movie-id mapping
mappings = torch.load('mappings.pt')
user_id_mapping = mappings['user_id_mapping']
movie_id_mapping = mappings['movie_id_mapping']
mapping = mappings['mapping']
# user_id_mapping = {id:i for i, id in enumerate(df['User'].unique())}
# movie_id_mapping = {id:i for i, id in enumerate(df['Movie'].unique())}
user_id_reverse_mapping = {v: k for k, v in user_id_mapping.items()}
movie_id_reverse_mapping = {v: k for k, v in movie_id_mapping.items()}

# Use mapping to get better ids
df['User'] = df['User'].map(user_id_mapping)
df['Movie'] = df['Movie'].map(movie_id_mapping)

##### Combine both datasets to get movies with metadata
# Preprocess metadata
tmp_metadata = movie_metadata.copy()
tmp_metadata.index = tmp_metadata.index.str.lower()

# Preprocess titles
tmp_titles = movie_titles.drop('Year', axis=1).copy()
tmp_titles = tmp_titles.reset_index().set_index('Name')
tmp_titles.index = tmp_titles.index.str.lower()

# Combine titles and metadata
df_id_descriptions = tmp_titles.join(tmp_metadata).dropna().set_index('Id')
df_id_descriptions['overview'] = df_id_descriptions['overview'].str.lower()
del tmp_metadata,tmp_titles

# Filter all ratings with metadata
df_hybrid = df.drop('Date', axis=1).set_index('Movie').join(df_id_descriptions).dropna().drop('overview', axis=1).reset_index().rename({'index':'Movie'}, axis=1)
print(df_id_descriptions.head())

# mapping = {id:i for i, id in enumerate(df_id_descriptions.index)}

print('Loading precalculated overview vectors')
sentence_vectors = torch.load('overview_vectors.pt', map_location=lambda storage, loc: storage)

print('Loading pretrained model')
import json
from model import DeepRecommender

with open('config.json') as config_fp:
  config = DotDict(json.load(config_fp))
model = DeepRecommender(config)
try:
  model.load_state_dict(
    torch.load('model.pt', map_location=lambda storage, loc: storage)['state_dict']
  )
except:
  print('Error while loading pretrained model')

def get_recommendation_existing_user():
  print('Type in an existing user ID')
  user_id = input()
  if (user_id == CANCEL_KEY):
    return
  else:
    try:
      user_id = int(user_id)
      user_idx = user_id_mapping[user_id]
    except:
      return

    all_movies = (df[df['User'] == user_idx])
    all_movies['Title'] = all_movies.apply(lambda row: movie_titles.loc[int(row['Movie'])]['Name'], axis=1)
    print(f'Movies rated by user {user_id}')
    print(all_movies[['Title', 'Rating']].sort_values(by=['Rating'], ascending=False))
    
    rated_movies = all_movies['Movie'].values
    unrated_movies = [id for id in range(0, config.movies) 
      if id in movie_id_reverse_mapping.keys() and
      movie_id_reverse_mapping[id] not in rated_movies
    ]

    results = []
    batch_size = 32
    # print(unrated_movies)
    for start_ix in tqdm(range(0, len(unrated_movies), batch_size)):
      batch_idxs = unrated_movies[start_ix:min(start_ix + batch_size, len(unrated_movies) - 1)]
      batch_idxs = [id for id in batch_idxs if id in movie_id_reverse_mapping.keys()]
      batch_ids = [movie_id_reverse_mapping[id] for id in batch_idxs]
      batch_size = len(batch_idxs)
      # print(batch_idxs)
      if batch_size > 0:
        batch_user = [user_idx] * batch_size

        indices = [mapping[id] if id in mapping else -1 for id in batch_ids]
        overview_vecs = torch.stack([
          sentence_vectors[id]
          if id > -1
          else torch.zeros(768).float()
          for id in indices
        ])
        
        with torch.no_grad():
          batch_result = model(
            torch.LongTensor(batch_user),
            torch.LongTensor(batch_idxs), 
            overview_vecs
          )
          results.extend(batch_result)

    results = torch.stack(results).squeeze(1)
    # print(results.size())
    sorted_scores, sorted_indices = torch.sort(results, dim=-1, descending=True)
    unrated_movies = torch.LongTensor(unrated_movies)[sorted_indices]

    # print(sorted_indices)
    print('\n\n=== Top 10 Recommendations: ===\n\n')
    num_recommends = min(len(unrated_movies), 10)
    for ix in range(num_recommends):
      movie_id = movie_id_reverse_mapping[int(unrated_movies[ix])]
      print(movie_titles[movie_titles['Id'] == movie_id])
      print(f'Score: {sorted_scores[ix]}\n\n')


def get_single_score():
  print('Type in an existing user ID')
  user_id = input()
  if (user_id == CANCEL_KEY):
    return
  else:
    try:
      user_id = int(user_id)
      user_id = user_id_reverse_mapping[user_id]
    except:
      return

  print('Type in an existing movie ID')
  movie_id = input()
  if (movie_id == CANCEL_KEY):
    return
  else:
    try:
      movie_id = int(movie_id)
      movie_id = movie_id_reverse_mapping[movie_id]
    except:
      return

  overview_vec = sentence_vectors[movie_id].unsqueeze(0)
  with torch.no_grad:
    score = model(
      torch.LongTensor([user_id]),
      torch.LongTensor([movie_id]),
      overview_vec
    )
  print(f'Calculated score: {score[0]}')

def add_user():
  import torch
  import torch.nn as nn
  config.users += 1
  user_emb_weights = model.user_emb.weight
  # new_weights = torch.randn(1, config.user_embedding_size)
  new_weights = torch.zeros(1, config.user_embedding_size)
  new_emb_weights = torch.cat([user_emb_weights, new_weights], dim=0)
  print(f'Old users embedding weights: {user_emb_weights.size()}')
  print(f'New users embedding weights: {new_emb_weights.size()}')
  model.user_emb = nn.Embedding.from_pretrained(new_emb_weights)
  
  return config.users - 1

def add_movie(title, overview):
  import torch
  import torch.nn as nn
  config.movies += 1
  new_movie_id = config.movies - 1

  movie_emb_weights = model.movie_emb.weight
  # new_weights = torch.randn(1, config.movie_embedding_size)
  new_weights = torch.zeros(1, config.movie_embedding_size)
  new_emb_weights = torch.cat([movie_emb_weights, new_weights], dim=0)
  print(f'Old users embedding weights: {movie_emb_weights.size()}')
  print(f'New users embedding weights: {new_emb_weights.size()}')
  model.movie_emb = nn.Embedding.from_pretrained(new_emb_weights)

  from create_overview_vectors import get_sentence_vector
  overview_vec = get_sentence_vector([overview])[0].unsqueeze(0)
  sentence_vectors = torch.cat([sentence_vectors, overview_vec], axis=0)
  mapping[new_movie_id] = sentence_vectors.size(0) - 1

  return new_movie_id


if __name__ == "__main__":
  new_user_id = add_user()
  while True:
    print(f"""
    A test user has been created. User ID: {new_user_id}.

    Choose one of the following commands (or type 'exit'):
    
    1. Get recommendations for an existing user
    2. Get single score for user_id, movie_id pair

    Test user commands:

    2. Add a movie title to the test user
    """)
    command = input()
    if (command == '1'):
      get_recommendation_existing_user()

    if (command == '2'):
      get_single_score()

    elif (command == 'exit'):
      exit()

    else:
      pass