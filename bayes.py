import datasetTest

artists = ["beyonce-knowles", "50-cent", "eazy-e", "casey-veggies", "fetty-wap",
    "flatbush-zombies", "bas", "frank-ocean", "grandmaster-flash", "childish-gambino", "clipse",
    "big-l", "aloe-blacc", "eminem", "future", "flobots", "david-banner"]

n = 3
ignore = []
pseudocount = 0.1

def extractWordFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
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
        # END_YOUR_CODE
    return extract

featureExtractor = extractWordFeatures(n)

train_x = []
train_y = []
test_x = []
test_y = []

original_list = datasetTest.my_list
count = 0 #use count to add every 5th example to test_examples (as opposed to train_examples)
for example_string in original_list:
    artist = example_string.split(" ")[0]
    lyrics = (" ").join(example_string.split(" ")[1:])
    if count < 4:
        train_x.append(featureExtractor(lyrics))
        train_y.append(artists.index(artist))
        count += 1
    else: 
        test_x.append(featureExtractor(lyrics))
        test_y.append(artists.index(artist))
        count = 0

#increment counts
gram_counts = []
for i in range(len(artists)):
	gram_counts.append({})
for i in range(len(train_x)):
	artist_count = gram_counts[train_y[i]]
	for gram in train_x[i]:
		value = train_x[i][gram]
		if gram in artist_count:
			artist_count[gram] += value
		else:
			artist_count[gram] = value

grams = set()
for x in train_x + test_x:
	for gram in x:
		grams.add(gram)

#laplace smooth
for gram_count in gram_counts:
	for gram in grams:
		if gram in gram_count:
			gram_count[gram] += pseudocount
		else:
			gram_count[gram] = pseudocount

#normalize
for gram_count in gram_counts:
	total = 0
	for gram in gram_count:
		total += gram_count[gram]
	for gram in gram_count:
		gram_count[gram] = gram_count[gram] * 1.0 / total

errors = 0
for i in range(len(test_x)):
	x_grams = test_x[i]
	max_artist_index = 0
	max_artist_prob = float('-inf')
	for artist_index in range(len(artists)):
		prob = 1.0
		artist_gram_count = gram_counts[artist_index]
		for gram in x_grams:
			prob *= artist_gram_count[gram]
		if prob > max_artist_prob:
			max_artist_prob = prob
			max_artist_index = artist_index
	if test_y[i] != max_artist_index:
		errors += 1
	else:
		print 'correctly predicted artist '+str(artists[max_artist_index])+' with probability '+str(max_artist_prob)
print 'naive bayes predicted artists in test set with '+str(1-(1.0*errors/len(test_x)))+' accuracy'
print len(test_x)
print len(train_x)
