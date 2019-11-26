import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import torch.utils.data as data_utils
from utils import DotDict


# DEFAULT_CONFIG = DotDict({
#     'users': len(user_id_mapping),
#     'movies': len(movie_id_mapping),

#     'user_embedding_size': 64,
#     'movie_embedding_size': 64,
#     'metadata_size': 128,
    
#     'hidden_size': 256
# })

# with open('config.json', 'w') as f:
#     json.dump(DEFAULT_CONFIG, f)

class DeepRecommender(nn.Module):
    
    def __init__(self, config):
        super(DeepRecommender, self).__init__()
        
        self.user_emb = nn.Embedding(config.users, config.user_embedding_size)
        self.movie_emb = nn.Embedding(config.movies, config.movie_embedding_size)
        
        self.dim_reduction = nn.Linear(768, config.metadata_size)
        
        self.output = nn.ModuleList([
            nn.Linear(config.user_embedding_size + config.movie_embedding_size + config.metadata_size, \
                                config.hidden_size),
            nn.Dropout(.2),
            nn.Linear(config.hidden_size, 1)
        ])
        
    def forward(self, user, movie, metadata):
        # print(user.size())
        # print(movie.size())
        # print(metadata.size())
        user_emb = self.user_emb(user)
        movie_emb = self.movie_emb(movie)
        metadata = self.dim_reduction(metadata)
        x = torch.cat([user_emb, movie_emb, metadata], dim=-1)
        
        for module in self.output:
            x = module(x)
        
        return x
