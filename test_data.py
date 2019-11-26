import torch

import pandas as pd
from tqdm import tqdm

from collections import deque

# Load a movie metadata dataset
movie_metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
# Remove the long tail of rarely rated moves
movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)

print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))

# Load data for all movies
movie_titles = pd.read_csv('./netflix-prize-data/movie_titles.csv', 
                           encoding = 'ISO-8859-1', 
                           header = None, 
                           names = ['Id', 'Year', 'Name']).set_index('Id')

print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))
# movie_titles.sample(5)

# Load single data-file
df_raw = pd.read_csv('./netflix-prize-data/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

# Find empty rows to slice dataframe for each movie
tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

# Shift the movie_indices by one to get start and endpoints of all movies
shifted_movie_indices = deque(movie_indices)
shifted_movie_indices.rotate(-1)


# Gather all dataframes
user_data = []

# Iterate over all movies
for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
    
    # Check if it is the last movie in the file
    if df_id_1<df_id_2:
        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
    else:
        tmp_df = df_raw.loc[df_id_1+1:].copy()
        
    # Create movie_id column
    tmp_df['Movie'] = movie_id
    
    # Append dataframe to list
    user_data.append(tmp_df)

# Combine all dataframes
df = pd.concat(user_data)
del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id

print('Shape User-Ratings:\t{}'.format(df.shape))

while True:
    try:
        print('Enter user id')
        user_id = input()

        if user_id == 'exit':
            exit()

        all_movies = (df[df['User'] == user_id])
        all_movies['Title'] = all_movies.apply(lambda row: movie_titles.loc[int(row['Movie'])]['Name'], axis=1)
        print(f'Movies rated by user {user_id}')
        print(all_movies[['Title', 'Rating']].sort_values(by=['Rating'], ascending=False))
    except KeyboardInterrupt:
        exit()
