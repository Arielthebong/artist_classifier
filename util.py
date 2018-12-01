import datasetTest
import dataset25Artists
n = 2
ignore = []

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

def extractWordFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
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
    return extract

train_x = []
train_y = []
test_x = []
test_y = []

train_examples = []
test_examples = []

featureExtractor = extractWordFeatures(n)

original_list = dataset25Artists.my_list
count = 0 #use count to add every 5th example to test_examples (as opposed to train_examples)
for example_string in original_list:
    artist = example_string.split(" ")[0]
    if artist not in artists:
        artists.append(artist)
    lyrics = (" ").join(example_string.split(" ")[1:])
    if count < 4:
        train_examples.append([artist, lyrics])
        train_x.append(featureExtractor(lyrics))
        train_y.append(artists.index(artist))
        count += 1
    else: 
        test_examples.append([artist, lyrics])
        test_x.append(featureExtractor(lyrics))
        test_y.append(artists.index(artist))
        count = 0