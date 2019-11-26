import torch

import pandas as pd
from tqdm import tqdm

from transformers.modeling_distilbert import *
from transformers.tokenization_distilbert import *
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').cuda()

from keras_preprocessing.sequence import pad_sequences

def get_sentence_vector(input_texts):
    input_ids = torch.LongTensor(pad_sequences([
        tokenizer.encode(input_text, add_special_tokens=True)
        for input_text in input_texts
    ], maxlen=512)).cuda()
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states[:,0,:].cpu()

if __name__ == "__main__":
    # Load a movie metadata dataset
    movie_metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
    # Remove the long tail of rarly rated moves
    movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)

    print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))

    batch_size=128
    train_tfidf = []
    # Iterate over all movie-ids and save the tfidf-vector
    values = movie_metadata['overview'].values
    for start_idx in tqdm(range(0, len(values), batch_size)):
        sentences = [
            str(value)
            for value in values[start_idx:min(start_idx+batch_size, len(values) - 1)]
        ]
        train_tfidf.extend(get_sentence_vector(sentences))

    torch.save(train_tfidf, 'overview_vectors.pt')
