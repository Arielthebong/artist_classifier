import datasetTest
import dataset25Artists
import newDataset_25_NewArtists_1
from random import shuffle
import gensim 
from gensim.models import Word2Vec
import numpy as np

n = 3
ignore = ['the', 'a', 'an']

artists = []

use_word_embedding_model = False
word_embedding_dim = 100

print_correct_counts = False
print_incorrect_pairs = False
print_false_positive_rates = False

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def featureExtractor(m, x):
    x_list = [i.lower() for i in x.split(' ')]
    result0 = []
    result = {}
    for iw in ignore:
        if iw in x_list:
            x_list.remove(iw)
    for i in range(len(x_list)-n+1):
        gram = tuple(x_list[i:i+m])
        gram_string = ' '.join(gram)
        result0.append(gram_string)
        if gram in result:
            result[gram] += 1
        else:
            result[gram] = 1
    return result0, result

train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []

def create_sets(m):
    original_list = newDataset_25_NewArtists_1.my_list
    #shuffle(original_list)
    count = 0 #use count to add every 5th example to test_examples (as opposed to train_examples)
    song_tokens = []
    for example_string in original_list:
        artist = example_string.split(" ")[0]
        if artist not in artists:
            artists.append(artist)
        lyrics = (" ").join(example_string.split(" ")[1:])
        gram_tokens, song_as_dict = featureExtractor(m, lyrics)
        song_tokens.append(gram_tokens)
        if count < 7:
            train_x.append(song_as_dict)
            train_y.append(artists.index(artist))
        elif count == 7:
            val_x.append(song_as_dict)
            val_y.append(artists.index(artist))
        else:
            test_x.append(song_as_dict)
            test_y.append(artists.index(artist))
        count += 1
        if count > 9:
            count = 0

    return song_tokens, (train_x, train_y, val_x, val_y, test_x, test_y)

train_x, train_y, val_x, val_y, test_x, test_y = create_sets(n)[1]

def getWord2VecDict():
    model = Word2Vec.load('25artists_'+str(n)+'gram.model')
    WordVectorz=dict(zip(model.wv.index2word,model.wv.vectors))
    return WordVectorz

def songAsWord2VecAverage(lyrics_as_dict):
    model_dict = getWord2VecDict()
    avg_this = []
    for gram, count in lyrics_as_dict.items():
        for i in range(count):
            gram_string = ' '.join(gram)
            if gram_string in model_dict:
                model_result = model_dict[gram_string]
                avg_this.append(model_result)
    if len(avg_this) == 0:
        print 'incoming 0 vector'
        return [0]*word_embedding_dim
    result = np.around(np.nanmean(avg_this, axis=0), 4)
    return result

def x_sets_as_word_embeddings(train_x, val_x, test_x):
    print 'rewriting all x sets with word embedding model'
    new_train_x = []
    for x in train_x:
        new_train_x.append(songAsWord2VecAverage(x))
    train_x = new_train_x

    new_val_x = []
    for x in val_x:
        new_val_x.append(songAsWord2VecAverage(x))
    val_x = new_val_x

    new_test_x = []
    for x in test_x:
        new_test_x.append(songAsWord2VecAverage(x))
    test_x = new_test_x

    return train_x, val_x, test_x