from util import *
import gensim 
from gensim.models import Word2Vec


for i in range(1,4):
	print 'creating word2vec model for n='+str(i)
	song_tokens = create_sets(i)[0]
	model = gensim.models.Word2Vec(song_tokens)
	model.save('25artists_'+str(i)+'gram.model')