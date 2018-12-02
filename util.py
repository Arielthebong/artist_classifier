import datasetTest
import dataset25Artists
import newDataset_25_NewArtists_1
from random import shuffle

n = 1
ignore = ['the', 'a']

artists = []

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

def featureExtractor(x):
    x_list = x.split(' ')
    result = {}
    for i in range(len(x_list)-n+1):
        gram = tuple(x_list[i:i+n])
        use = True
        for iw in ignore:
        	if iw in gram:
        		use = False
        if use:
            if gram in result:
                result[gram] += 1
            else:
                result[gram] = 1
    return result

train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []

original_list = newDataset_25_NewArtists_1.my_list
#shuffle(original_list)
count = 0 #use count to add every 5th example to test_examples (as opposed to train_examples)
for example_string in original_list:
    artist = example_string.split(" ")[0]
    if artist not in artists:
        artists.append(artist)
    lyrics = (" ").join(example_string.split(" ")[1:])
    if count < 7:
        train_x.append(featureExtractor(lyrics))
        train_y.append(artists.index(artist))
    elif count == 7:
        val_x.append(featureExtractor(lyrics))
        val_y.append(artists.index(artist))
    else:
        test_x.append(featureExtractor(lyrics))
        test_y.append(artists.index(artist))
    count += 1
    if count > 9:
        count = 0