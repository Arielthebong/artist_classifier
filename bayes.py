from util import *
from collections import defaultdict
from random import choice

train_x += val_x
train_y += val_y

#for laplace smoothing; only applies to gram probs not artist probs
pseudocount = [0.05]

#include probabilty of track being by artist independent of lyrics in probabilistic inference
include_artist_prob = False

#where probabilities will be stored
gram_counts = []
artist_counts = [0]*len(artists)

artist_correct = [0] * len(artists)
artist_totals = [0] * len(artists)
artist_false_positives = [0] * len(artists)

def tune_counts():
	print 'tuning counts'
	gram_counts = []
	artist_counts = [0]*len(artists)

	#increment counts
	for i in range(len(artists)):
		if use_word_embedding_model:
			gram_counts.append([defaultdict(int) for _ in range(word_embedding_dim)])
		else:
			gram_counts.append(defaultdict(int))

	for i in range(len(train_x)):
		artist_count = gram_counts[train_y[i]]
		artist_counts[train_y[i]] += 1
		if not use_word_embedding_model:
			for gram in train_x[i]:
				value = train_x[i][gram]
				artist_count[gram] += value
		else:
			for i, value in enumerate(train_x[i]):
				artist_count[i][value] += 1


	#laplace smooth

	if not use_word_embedding_model:
		grams = set()
		for x in train_x + test_x:
			for gram in x:
				grams.add(gram)
		for gram_count in gram_counts:
			for gram in grams:
				if gram in gram_count:
					gram_count[gram] += pseudocount[0]
				else:
					gram_count[gram] = pseudocount[0]
	else:
		potential_values_by_dim = []
		for i in range(word_embedding_dim):
			potential_values_by_dim.append([])
		for x in train_x + test_x:
			for i in range(word_embedding_dim):
				potential_values_by_dim[i].append(x[i])
		for artists_dim_counts in gram_counts:
			for i, dim_count in enumerate(artists_dim_counts):
				for potential_value in potential_values_by_dim[i]:
					dim_count[potential_value] += pseudocount[0]


	#normalize

	if not use_word_embedding_model:
		new_gram_counts = []
		for gram_count in gram_counts:
			new_gram_count = {}
			total = 0
			for gram in gram_count:
				total += gram_count[gram]
			for gram in gram_count:
				new_gram_count[gram] = gram_count[gram] * 1.0 / total
			new_gram_counts.append(new_gram_count)
		gram_counts = new_gram_counts

		total = sum(artist_counts)
		new_artist_counts = []
		for artist_count in artist_counts:
			new_artist_counts.append(1.0 * artist_count / total)
		artist_counts = new_artist_counts
	else:
		new_gram_counts = []
		for artists_dim_counts in gram_counts:
			new_artists_dim_counts = []
			for dim_count in artists_dim_counts:
				new_dim_count = defaultdict(int)
				total = 0
				for value, count in dim_count.items():
					total += count
				for value, count in dim_count.items():
					new_dim_count[value] = count * 1.0 / total
				new_artists_dim_counts.append(new_dim_count)
			new_gram_counts.append(new_artists_dim_counts)
		gram_counts = new_gram_counts

		#todo: artists_counts normalization

	return artist_counts, gram_counts

#returns accuracy rate predicting y from x
def evaluatePredictor(x, y, print_inaccurate_pairs=False, update_correct_counts=False, update_false_positive_counts=False):
    randomized_prediction_count = 0
    incorrect_pairs = defaultdict(int)
    errors = 0
    for i in range(len(x)):
		x_grams = x[i]
		max_artist_index = -1
		max_artist_prob = float('-inf')
		for artist_index in range(len(artists)):
			prob =  1.0
			if include_artist_prob:
				prob *= artist_counts[artist_index]
			if use_word_embedding_model:
				artists_dim_counts = gram_counts[artist_index]
				for j in range(word_embedding_dim):
					prob *= artists_dim_counts[j][x_grams[j]]
			else:
				artist_gram_count = gram_counts[artist_index]
				for gram in x_grams:
					gram_prob = artist_gram_count[gram]
					prob *= gram_prob
			if prob > max_artist_prob:
				max_artist_prob = prob
				max_artist_index = artist_index
		assert max_artist_index != -1
		if max_artist_prob == 0:
			randomized_prediction_count += 1
			max_artist_index = choice(range(len(artists)))
		if y[i] != max_artist_index:
			if update_false_positive_counts:
				artist_false_positives[max_artist_index] += 1
			errors += 1
			pair = [artists[max_artist_index], artists[y[i]]]
			pair.sort()
			incorrect_pairs[tuple(pair)] += 1
		else:
			if update_correct_counts:
				artist_correct[max_artist_index] += 1
		if update_correct_counts:
			artist_totals[y[i]] += 1
    if print_inaccurate_pairs:
        d_view = [ (v,k) for k,v in incorrect_pairs.iteritems() ]
        d_view.sort(reverse=True) # natively sort tuples by first element
        print d_view
    print str(randomized_prediction_count) + ' predictions were made at random'
    return (1.0*errors/len(x))
		#else:
			#print 'correctly predicted artist '+str(artists[max_artist_index])+' with probability '+str(max_artist_prob)

if use_word_embedding_model: train_x, val_x, test_x = x_sets_as_word_embeddings(train_x, val_x, test_x)
artist_counts, gram_counts = tune_counts()
print 'naive bayes predicted artists in train set with '+str(evaluatePredictor(train_x, train_y))+' error'
print 'naive bayes predicted artists in test set with '+str(evaluatePredictor(test_x, test_y, print_incorrect_pairs, print_correct_counts, print_false_positive_rates))+' error'

if print_correct_counts:
	print 'artists in order of decreasing percentage of appearances in test set that are correctly predicted:'
	sort_this = []
	for i in range(len(artists)):
		sort_this.append((100.0*artist_correct[i]/artist_totals[i],artists[i]))
	sort_this.sort(reverse=True)
	for ranked_artist in sort_this:
		print ranked_artist[1]+': '+str(ranked_artist[0])+'%'

if print_false_positive_rates:
	print 'artists in order of decreasing number of false positives (where that artist was output of the inaccurate prediction):'
	sort_this = []
	for i in range(len(artists)):
		sort_this.append((artist_false_positives[i],artists[i]))
	sort_this.sort(reverse=True)
	for ranked_artist in sort_this:
		print ranked_artist[1]+': '+str(ranked_artist[0])


