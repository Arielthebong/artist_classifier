from util import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import sklearn.feature_extraction

numIters = 20
eta = 0.01
lamb = 0.1

weights_all = []

def learnPredictor(trainExamples, featureExtractor, numIters, eta, lamb):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, returns the weight vector (sparse
    feature vector) learned.
    '''
    weights = {}  # feature => weight

    for i in range(numIters):
        for c in trainExamples:
            x = c[1]
            phi = featureExtractor(x)
            y = c[0]
            gradient = 0
            margin = dotProduct(phi, weights)*y
            if margin < 1:
                gradient = -sum(phi.values())*y
            for f in phi:
                for j in range(phi[f]):
                    if f in weights:
                        weights[ f] -= eta * (gradient + lamb * weights[f])
                    else:
                        weights[f] = - eta * gradient
    return weights

#returns predicted artist
def predict_artist(lyrics):
    highest_score_index = 0
    highest_score = float('-inf')
    for i, weights in enumerate(weights_all):
        phi = featureExtractor(lyrics)
        score = dotProduct(phi, weights)
        if score > highest_score:
            highest_score_index = i
            highest_score = score
    artist = artists[highest_score_index]
    #print 'highest score: '+str(highest_score)+ ' for artist '+str(artist)
    return artist

def evaluatePredictors(examples):
    error = 0
    for example in examples:
        predicted_artist = predict_artist(example[1])
        true_artist = example[0]
        if predicted_artist != true_artist:
            #print 'prediction was incorrect'
            error += 1
        #else:
            #print 'prediction was correct'
    return 1.0 * error / len(examples)

'''
print "running using sklearn's SGDClassifier..."

#for other library implementations
def evaluatePredictor(predictor_name, set_type, true_y, predicted_y):
    errors = 0
    for i in range(len(true_y)):
        if true_y[i] != predicted_y[i]:
            errors += 1.0
    print predictor_name+' predicted artists in '+set_type+' set with '+str(1.0-errors/len(true_y))+' percent accuracy'

v = sklearn.feature_extraction.DictVectorizer()
total_x = v.fit_transform(train_x+test_x)
split_index = len(train_x)
train_x = total_x[:split_index]
test_x = total_x[split_index:]

artist_predictor = SGDClassifier(loss='hinge', max_iter=numIters, learning_rate='constant', eta0=eta, penalty='l2').fit(train_x, train_y)
predicted_artist_indices_train = artist_predictor.predict(train_x)
predicted_artist_indices_test = artist_predictor.predict(test_x)
evaluatePredictor('sklearn', 'train', train_y, predicted_artist_indices_train)
evaluatePredictor('sklearn', 'test', test_y, predicted_artist_indices_test)


print 'running using custom implementation...'
'''
for i in range(len(artists)):
    print 'training for artist '+str(artists[i])

    #edit examples for particular artist (+1 if by artist i, -1 if not)
    new_train_examples = []
    for train_example in train_examples:
        new_example = []
        if train_example[0] == artists[i]:
            new_example.append(1)
        else:
            new_example.append(-1)
        new_example.append(train_example[1])
        new_train_examples.append(new_example)

    weights_all.append(learnPredictor(new_train_examples, featureExtractor, numIters, eta, lamb))
print 'SGD predicted artist on train examples with '+str(1.0-evaluatePredictors(train_examples))+' accuracy'
print 'SGD predicted artist on test examples with '+str(1.0-evaluatePredictors(test_examples))+' accuracy'


