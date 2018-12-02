from util import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import sklearn.feature_extraction
from matplotlib import pyplot as plt

numIters = 20
eta = 0.01
lamb = 0.5
weights_all = [{} for _ in range(len(artists))]

def learnPredictor(weights, x, y, featureExtractor, eta, lamb):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, returns the weight vector (sparse
    feature vector) learned.
    '''
    #weights = {}  # feature => weight

    for j in range(len(x)):
        phi = x[j]
        y_i = y[j]
        gradient = 0
        margin = dotProduct(phi, weights)*y_i
        if margin < 1:
            gradient = -sum(phi.values())*y_i
        for f in phi:
            for j in range(phi[f]):
                if f in weights:
                    weights[f] -= eta * (gradient + lamb * weights[f])
                else:
                    weights[f] = - eta * gradient

#returns predicted artist
def predict_artist(phi):
    highest_score_index = 0
    highest_score = float('-inf')
    for i, weights in enumerate(weights_all):
        score = dotProduct(phi, weights)
        if score > highest_score:
            highest_score_index = i
            highest_score = score
    #print 'highest score: '+str(highest_score)+ ' for artist '+str(artists[highest_score_index])
    return highest_score_index

def evaluatePredictors(x, y):
    error = 0
    for i in range(len(x)):
        predicted_y = predict_artist(x[i])
        true_y = y[i]
        if predicted_y != true_y:
            #print 'prediction was incorrect'
            error += 1
        #else:
            #print 'prediction was correct'
    return 1.0 * error / len(x)

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
train_errors = []
val_errors = []
test_errors = []
for j in range(numIters):
    print 'iteration '+str(j)
    for i in range(len(artists)):
        #print 'training for artist '+str(artists[i])

        #edit y for particular artist (+1 if by artist i, -1 if not)
        new_train_y = []
        for y_i in train_y:
            if y_i == i:
                new_train_y.append(1)
            else:
                new_train_y.append(-1)

        learnPredictor(weights_all[i], train_x, new_train_y, featureExtractor, eta, lamb)
    train_error = evaluatePredictors(train_x, train_y)
    train_errors.append(train_error)
    val_error = evaluatePredictors(val_x, val_y)
    val_errors.append(val_error)
    test_error = evaluatePredictors(test_x, test_y)
    test_errors.append(test_error)
    
print 'training error: '+str(train_error)
print 'val error: '+str(val_error)
print 'test error: '+str(test_error)
x = range(numIters)
plt.plot(x, train_errors, label='Training Error')
plt.plot(x, val_errors, label='Dev Error')
plt.plot(x, test_errors, label='Test Error')
plt.xticks(range(1, numIters+1))
plt.xlabel('iteration')
plt.ylabel('error rate')
plt.title('Error Rate over Iterations for SVM with N='+str(n))
plt.legend()
plt.show()

