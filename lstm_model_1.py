import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
#%matplotlib inline
from typing import Set, Any

'''
def remove_rows_except(df, rows_to_keep: Set[Any]):
    rows_total: Set[Any] = set(df.rows)
    diff: Set[Any] = rows_total - rowsToKeep
    df.drop(diff, inplace = True)
'''

# read in CSV, replace blank lyrics with empty string
#df = pd.read_csv('lyrics.csv', converters={'lyrics' : lambda x: "" if x == np.nan else x})

df = pd.read_csv('lyrics.csv')

df.dropna(how='any', inplace=True) #drop all blank rows
#df.dropna(subset=['lyrics'])

#print(df.head())

df.drop(['index', 'song', 'year', 'genre'],axis=1,inplace=True)


old_25_Artists = ["beyonce-knowles", "50-cent", "eazy-e", "casey-veggies", "fetty-wap", "flatbush-zombies", "bas", "frank-ocean", "grandmaster-flash", "childish-gambino", "clipse", "big-l", "aloe-blacc", "eminem", "future", "flobots", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"]

new_25_Artists = ["beyonce-knowles", "50-cent", "eazy-e", "asap-ferg", "fetty-wap", "danny-brown", "flo-rida", "frank-ocean", "action-bronson", "childish-gambino", "clipse", "b-o-b", "aloe-blacc", "eminem", "future", "akon", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"]

union_31_Artists = ["beyonce-knowles", "50-cent", "eazy-e", "asap-ferg", "fetty-wap", "danny-brown", "flo-rida", "frank-ocean", "action-bronson", "childish-gambino", "clipse", "b-o-b", "aloe-blacc", "eminem", "future", "akon", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"] + ["beyonce-knowles", "50-cent", "eazy-e", "casey-veggies", "fetty-wap", "flatbush-zombies", "bas", "frank-ocean", "grandmaster-flash", "childish-gambino", "clipse", "big-l", "aloe-blacc", "eminem", "future", "flobots", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"]
union_31_Artists = list(set(union_31_Artists))

more_than_100_artists = ['garbage', 'gloria-trevi', 'the-clash', 'fred-hammond', 'america', 'eminem', 'george-harrison', 'bobby-darin', 'alabama', 'girls-aloud', 'a-ha', 'dashboard-confessional', 'eros-ramazzotti', 'beyonce-knowles', 'deep-purple', 'against-me', 'clay-aiken', 'bodyjar', 'disney', 'cliff-richard', 'cheap-trick', 'divine-comedy', 'david-allan-coe', 'danny-elfman', 'dr-feelgood', 'gerry-rafferty', 'dream-theater', 'edaiaoidee', 'dannii-minogue', 'dokken', 'anvil', 'elliott-smith', 'chemical-brothers', 'elvis-presley', 'gary-numan', 'boyzone', 'dionne-warwick', 'alkaline-trio', 'devo', 'adrian-belew', 'bob-mould', 'devendra-banhart', 'ben-folds', 'billy-gilman', 'diana-ross-the-supremes', 'enrique-iglesias', 'barry-manilow', 'brooks-dunn', 'arcade-fire', 'bizzy-bone', 'depeche-mode', 'dwight-yoakam', 'boehse-onkelz', 'cat-stevens', 'etta-james', 'cat-power', 'buddy-holly', 'the-cult', 'don-gibson', 'glenn-hughes', 'chevelle', 'cam-ron', 'deftones', 'evanescence', 'al-green', 'chief-keef', 'glen-campbell', 'dixie-chicks', 'conway-twitty', 'e-40', 'anathema', 'gilbert-osullivan', 'the-corrs', 'church', 'bjrthrk', 'blink-182', 'doro-pesch', 'death-cab-for-cutie', 'exodus', 'afi', 'chamillionaire', 'game', 'chesnutt-mark', 'bad-religion', 'carpenter-mary-chapin', 'badly-drawn-boy', 'cranberries', 'earth-wind-fire', 'ghostface-killah', 'bone-thugs-n-harmony', 'feeder', 'the-carpenters', 'black-sabbath', 'avett-brothers', 'ernest-tubb', 'dr-hook', 'chris-de-burgh', 'dead-milkmen', 'the-conells', 'flotsam-and-jetsam', 'foghat', 'the-alan-parsons-project', 'ella-fitzgerald', 'dr-dre', 'april-wine', 'andrea-bocelli', 'echo-the-bunnymen', 'chumbawamba', 'gipsy-kings', 'chet-baker', 'buck-owens', 'fury-in-the-slaughterhouse', 'alexandre-desplat', 'brad-paisley', 'alejandro-sanz', 'frenzal-rhomb', 'billy-joel', 'disciple', 'alison-krauss', 'five-iron-frenzy', 'brainstorm', 'disturbed', 'golden-earring', 'black-clint', 'ginuwine', 'cradle-of-filth', 'atmosphere', 'chaka-khan', 'aeena', 'entombed', 'don-omar', 'foo-fighters', 'dolly-parton', 'edguy', 'ayumi-hamasaki', 'dmx', 'ariana-grande', 'bobby-bare', 'clint-black', 'adam-lambert', 'bette-midler', 'chris-brown', 'barbra-streisand', 'frank-sinatra', 'bill-evans', 'asia', 'amon-amarth', 'bon-jovi', 'drake', 'anouk', 'children', 'ayreon', 'ciara', 'donna-summer', 'electric-six', 'blue-rodeo', 'the-doors', 'adam-and-the-ants', 'camel', 'danzig', 'blues-traveler', 'demi-lovato', 'blind-guardian', 'charlie-louvin', 'david-guetta', 'the-gathering', 'belle-and-sebastian', 'gloria-estefan', 'calexico', 'ben-lee', 'brenda-lee', 'curren-y', 'cece-winans', 'ellie-goulding', 'cenk-r-lr-etin', 'al-stewart', 'avicii', 'big-country', 'crosby-stills-nash', 'beck', 'axxis', 'the-animals', 'face-to-face', 'babyface', 'everything-but-the-girl', 'charlotte-church', 'george-jones', 'glee-cast', 'fat-joe', 'dusty-springfield', 'charlie-daniels-band', 'cassidy', 'agoraphobic-nosebleed', 'frank-zappa', 'gerald-levert', 'carlos-vives', 'busta-rhymes', 'biohazard', 'debbie-gibson', 'd-12', 'alice-cooper', 'dead-moon', 'andre-nickatina', 'gary-allan', 'eightball-mjg', 'ashanti', 'billie-holiday', 'brave-combo', 'chris-tomlin', 'cyndi-lauper', 'brian-mcknight', 'beastie-boys', 'craig-david', '2-chainz', 'allman-brothers-band', 'audio-adrenaline', 'chris-isaak', 'george-michael', 'cher', 'frank-black', 'camper-van-beethoven', 'decemberists', 'descendents', 'counting-crows', 'collin-raye', 'the-cure', 'accept', 'crystal-lewis', 'blondie', 'all', 'donovan-leitch', 'birdman', 'anthrax', 'ben-harper', 'clutch', 'faron-young', 'alanis-morissette', 'enigma', 'esham', 'del-reeves', '10-cc', 'alan-jackson', 'ana-gabriel', 'bee-gees', 'duncan-sheik', 'eddy-arnold', 'cocteau-twins', 'ferlin-husky', 'enya', 'emmylou-harris', 'carrie-underwood', 'b-o-b', 'elvis-costello', 'chris-ledoux', 'black-label-society', 'bryan-ferry', 'diamond-rio', 'battiato-franco', 'bing-crosby', 'connie-smith', 'doris-day', 'avril-lavigne', 'air-supply', 'erasure', 'aimee-mann', 'aesop-rock', 'christmas-song', 'ed-sheeran', 'backstreet-boys', 'emma-forman', 'dri', 'chico-buarque', 'ac-dc', 'crduan-xshadows', '2pac', 'future', 'chris-rea', 'boney-m', 'dar-williams', 'george-strait', 'electric-light-orchestra', 'blake-shelton', 'bobby-valentino', 'doc-watson', 'axel-rudi-pell', 'eric-carmen', 'blue-system', 'bush', 'casualties', 'common', 'exploited', 'aretha-franklin', 'better-than-ezra', 'destiny-s-child', 'abba', 'frank-ocean', 'dean-martin', 'blur', 'amorphis', 'cypress-hill', 'faithless', 'bif-naked', 'childish-gambino', 'buzzcocks', 'elton-john', 'billy-bragg', 'the-black-keys', 'blue-october', 'dinosaur-jr', 'bonnie-raitt', 'britney-spears', 'dan-fogelberg', 'cannibal-corpse', 'gamma-ray', 'g-b-h', 'aerosmith', 'david-hasselhoff', 'epmd', 'b-b-king', '50-cent', 'bill-anderson', 'avant', 'cowboy-junkies', 'eurythmics', 'gino-vannelli', 'die-toten-hosen', 'bananarama', 'carole-king', 'bill-monroe', 'arctic-monkeys', 'dancing-with-the-stars', 'all-time-low', 'donovan', 'delirious', 'boyz-ii-men', 'fleetwood-mac', 'anberlin', 'carly-simon', 'bad-company', 'agathocles', 'erykah-badu', 'burt-bacharach', 'acid-drinkers', 'fabolous', 'fats-domino', 'beautiful-south', 'band', 'eels', 'collective-soul', 'dave-matthews-band', 'coheed-and-cambria', 'french-montana', 'bonnie-tyler', 'don-williams', 'cracker', 'gomez', 'alphaville', 'annihilator', 'chuck-berry', 'the-donnas', 'fall', 'glee', 'big-head-todd-and-the-monsters', 'europe', 'celtic-woman', 'david-bowie', 'deacon-blue', 'akon', 'avenged-sevenfold', 'garth-brooks', 'barclay-james-harvest', 'foreigner', 'genesis', 'the-doobie-brothers', 'cardigans', 'anastacia', 'apocalyptica', 'clannad', 'behemoth', 'bob-seger', '311', 'charley-pride', 'canibus', 'alejandro-fernandez', 'andy-williams', 'freddie-hart', 'beach-boys', 'adam-sandler', 'ani-difranco', 'brian-wilson', 'ben-folds-five', 'flatt-and-scruggs', 'fish', 'daddy-yankee', 'american-idol', 'ana-belasn', 'black-eyed-peas', 'delta-goodrem', 'flaming-lips', 'fifth-harmony', 'christina-aguilera', 'arch-enemy', 'aaliyah', 'the-eagles', 'bow-wow', 'ataris', 'beyonce', 'bryan-adams', 'b-g', 'dir-en-grey', 'coldplay', 'big-sean', 'charlie-landsborough', 'buckcherry', 'clarks', 'ace-hood', 'alice-in-chains', 'billy-idol', 'godsmack', 'gary-moore', 'aaron-tippin', 'epica', 'bright-eyes', 'buddy-guy', 'dandy-warhols', 'biffy-clyro', 'dierks-bentley', 'brandy', 'edith-piaf', 'the-damned', 'barenaked-ladies', 'david-houston', 'big-bang', 'the-byrds', 'bouncing-souls', 'clifford-t-ward', 'amy-grant', 'ely-joe', 'girls-generation', 'bowling-for-soup', 'everclear', 'beatles', 'fall-out-boy', 'bathory', 'fishbone', 'faith-evans', 'daft-punk', 'david-gray', 'dark-tranquility', 'bob-dylan', 'david-crowder', 'alicia-keys', 'dottie-west', 'apulanta', 'christina-milian', 'all-4-one', 'diana-ross', 'boy-george', 'david-archuleta', 'brian-setzer', 'ash', 'andrew-lloyd-webber', 'front-line-assembly', 'ace-of-base', 'emerson-lake-palmer', 'blue-oyster-cult', 'celtic-thunder', 'eric-clapton', 'edwin-mccain', 'celine-dion', 'fear-factory', 'diana-krall', 'arlo-guthrie', 'armin-van-buuren', 'die-earzte', 'dropkick-murphys', 'big-k-r-i-t', 'cledus-t-judd', 'crash-test-dummies', 'converge', 'everly-brothers', 'afonso-zeca', 'bruce-hornsby', 'david-wilcox', 'aaron-neville', 'anti-flag', 'black-crowes', 'adaaeaaineay-iaidiia', 'bob-marley', 'cohen-leonard', 'def-leppard', 'dave-dudley', 'gangstarr', 'beenie-man', 'billy-ray-cyrus', 'clay-walker', 'dj-khaled', 'aygun-kaza-mova', 'funkmaster-flex', 'duran-duran', 'buckethead', 'drive-by-truckers', 'flo-rida', 'dio', 'david-byrne', 'de-la-soul', 'faith-no-more', 'faith-hill', 'bruce-springsteen', 'anne-murray', 'chicago', 'bob-marley-the-wailers', 'enslaved', 'brthhse-onkelz']

more_than_250_artists = ['elvis-costello', 'billie-holiday', 'bobby-bare', 'diana-ross', 'barbra-streisand', 'bill-anderson', 'bone-thugs-n-harmony', 'don-williams', 'boyz-ii-men', 'buck-owens', 'alice-cooper', '50-cent', 'andrea-bocelli', 'frank-sinatra', 'ernest-tubb', 'band', 'esham', 'dancing-with-the-stars', 'dusty-springfield', 'blondie', 'britney-spears', 'fabolous', 'dmx', 'ferlin-husky', 'die-toten-hosen', 'ella-fitzgerald', 'cliff-richard', 'barry-manilow', 'connie-smith', 'eddy-arnold', 'chris-brown', 'eric-clapton', 'the-byrds', 'aretha-franklin', 'gary-numan', 'bette-midler', 'chicago', 'ani-difranco', 'b-b-king', 'babyface', 'bob-dylan', 'bad-religion', 'dean-martin', 'busta-rhymes', 'chumbawamba', 'christina-aguilera', 'alan-jackson', 'erasure', 'beastie-boys', 'fleetwood-mac', 'charley-pride', 'electric-light-orchestra', 'the-cure', 'akon', 'elvis-presley', 'american-idol', 'beach-boys', 'ac-dc', 'david-bowie', 'bon-jovi', 'cher', 'alabama', '2pac', 'celine-dion', 'bee-gees', 'emmylou-harris', 'frank-zappa', 'e-40', 'beck', 'bruce-springsteen', 'carly-simon', 'fall', 'elton-john', 'conway-twitty', 'barenaked-ladies', 'chamillionaire', 'game', 'donna-summer', 'dolly-parton', 'eminem', 'beatles', 'drake']

all_artists = [artist for artist in df.artist]

old25artistsBool = False # Bool
new25artistsBool = False
union31artistsBool = False
onlyArtistsWith10plusSongsBool = False # Bool
onlyArtistsWith100plusSongsBool = False # Bool
onlyArtistsWith250plusSongsBool = True

if old25artistsBool:
    df = df[df.artist.isin(old_25_Artists)]
elif new25artistsBool:
    df = df[df.artist.isin(new_25_Artists)]
elif union31artistsBool:
    df = df[df.artist.isin(union_31_Artists)]
elif onlyArtistsWith10plusSongsBool:
    #df = df[df.artist.isin(selectedArtists)]
    artists_to_keep = []
    for artist in set(all_artists):
        if np.sum(df.artist == artist) >= 10:
            artists_to_keep.append(artist)
    print("Num artists with >= 10 songs: " + str(len(artists_to_keep)))
    print("ARTISTS WITH MORE THAN 10 SONGS: " + str(artists_to_keep))
elif onlyArtistsWith100plusSongsBool:
    df = df[df.artist.isin(more_than_100_artists)]
elif onlyArtistsWith250plusSongsBool:
    df = df[df.artist.isin(more_than_250_artists)]


'''
elif only_artists_with_10plus_songs:
    artists_to_drop = []
    for artist in set(all_artists):
        if np.sum(df.artist == artist) < 10:
            artists_to_drop.append(artist)
    df = df[~(df.artist.isin(artists_to_drop))]
elif only_artists_with_100plus_songs:
    artists_to_drop = []
    for artist in set(all_artists):
        if np.sum(df.artist == artist) < 100:
            artists_to_drop.append(artist)
elif only_artists_with_250plus_songs:
    print = False

    if(!print):
        artists_to_drop = []
        for artist in set(all_artists):
            if np.sum(df.artist == artist) < 250:
                artists_to_drop.append(artist)
    elif(print):
        artists_to_keep = []
        for artist in set(all_artists):
            if np.sum(df.artist == artist) >= 250:
                artists_to_keep.append(artist)
        print("Num artists with >= 250 songs: " + str(len(artists_to_keep)))
        print("ARTISTS WITH MORE THAN 250 SONGS: " + str(artists_to_keep))
    #df = df[~(df.artist.isin(artists_to_drop))]
'''



'''
artists_to_keep = ["beyonce-knowles", "50-cent", "eazy-e", "casey-veggies", "fetty-wap", "flatbush-zombies", "bas", "frank-ocean", "grandmaster-flash", "childish-gambino", "clipse", "big-l", "aloe-blacc", "eminem", "future", "flobots", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"]
#remove_rows_except(df, artists)
#all_artists = df.artist
#print("ALL ARTISTS: " + str(all_artists))

all_artists = [artist for artist in df.artist]

allIndicies = [i for i in range(len(all_artists))]

indiciesToKeep = []

for index, artist in enumerate(all_artists):
    if artist in artists_to_keep:
        indiciesToKeep.append(index)

indicies_of_artists_to_drop = list(set(allIndicies)-set(indiciesToKeep))

#all_artists = [artist for artist in df.artist]
#artists_to_drop = list(set(all_artists)-set(artists_to_keep))

#artistList = list(set(all_artists))

#print("ARTIST LIST: " + str(artistList))
#artists_to_keep = artists
#artists_to_drop = all_artists - artists_to_keep
#print("ARTISTS TO DROP: " + str(artists_to_drop))
df.drop(indicies_of_artists_to_drop, inplace=True)
'''

print(df.head())

print(df.info())

#print("ARTIST COUNT?: " + str(np.sum(df.artist == 'beyonce-knowles')))

# drop artists with fewer than 10 songs?
print(df.artist.value_counts())

#df.drop(['beyonce-knowles', 'eminem', '50-cent'], inplace=True)

'''
#artistNames = [x for x in np.unique(df.artist.values)]
#setArtistNames = set(artistNames)

#for name in setArtistNames

all_artists = [artist for artist in df.artist]

indiciesToDrop = []

seen = []


artists_to_drop = []
for artist in set(all_artists):
    #print("HERE 1")
    if np.sum(df.artist == artist) < 10:
        #print("DELETINGs")
        artists_to_drop.append(artist)
    #if np.random.randint(0, 200) == 100:
    #    break

#print("HERE 2!!!!!!!")
for index, artist in enumerate(all_artists):
    if artist in artists_to_drop:
        print("TRUE!")
        indiciesToDrop.append(index)

df.drop(indiciesToDrop, inplace=True)
    #if np.sum(artist == 'beyonce-knowles') < 5:
    #    indiciesToDrop.append
    #if artist in seen:
    #    print("ARTIST IN SEEN!")
    #else:
    #    seen.append(artist)
#    if np.sum(artist == 'beyonce-knowles') < 5:
#        indiciesToDrop.append


num_classes = len(np.unique(df.artist.values))

print("num_classes_2: " + str(num_classes))
'''


#sns.countplot(df.artist)
#plt.xlabel('Label')
#plt.title('Graph of songs per artist')
#plt.show()


num_classes = len(np.unique(df.artist.values))
print("num_classes: " + str(num_classes))


Y = df.artist
X = df.lyrics

le = LabelEncoder()
Y = le.fit_transform(Y)
#Y = Y.reshape(-1,1)  # WHAT is the purpose of this reshape???

Y = to_categorical(Y, num_classes=len(np.unique(df.artist.values)))


# BY Doing this, we test on a subset of ALL artists instead of our top 25
# maybe, to reduce # of artists, have it only look at artists who have more than
# 50 songs?
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.10, shuffle=True)

#X_val,X_test,Y_val,Y_test = train_test_split(X_testAndVal,Y_testAndVal,test_size=0.60, shuffle=True)

#print("X_train: " + str(X_train))

#print("Y_train: " + str(Y_train))

max_words = 100000 # this should probably be higher -- only keeps 1000000 most common words in dataset -- on the other hand, becomes very slow with higher values
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train) # problem -- need to get rid of blank rows of lyrics (which are represented as NaN) before calling this
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    #layer = Dense(256,name='FC1')(layer)
    # CHANGE -- 2 dense layers instead of 1
    layer = Dense(128,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(128,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.6)(layer) # this seems pretty high -- hyperparameter to tweak
    layer = Dense(num_classes,name='out_layer',activation='softmax')(layer)
    #layer = Dense(1,name='out_layer',activation='softmax')(layer)
    #layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=25,
          validation_split=0.10)#callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.000001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))