from util import *

pseudocount = 1

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
	max_artist_index = -1
	max_artist_prob = float('-inf')
	for artist_index in range(len(artists)):
		prob = 1.0
		artist_gram_count = gram_counts[artist_index]
		for gram in x_grams:
			gram_prob = artist_gram_count[gram]
			assert gram_prob != 0
			prob *= gram_prob
		if prob > max_artist_prob:
			max_artist_prob = prob
			max_artist_index = artist_index
	assert max_artist_index != -1
	if test_y[i] != max_artist_index:
		errors += 1
	else:
		print 'correctly predicted artist '+str(artists[max_artist_index])+' with probability '+str(max_artist_prob)
print 'naive bayes predicted artists in test set with '+str(1-(1.0*errors/len(test_x)))+' accuracy'
