import os
import pprint
import csv
import time
import pandas as pd
import numpy as np
import multiprocessing

import logging

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib

from matplotlib import pyplot as plt
import seaborn as sns

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models, utils, parsing
from gensim.models import KeyedVectors, Word2Vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from scipy.sparse import coo_matrix
from nltk.corpus import gutenberg

import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('gutenberg')

from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from keras.models import Model
# from keras.layers import Merge
from keras.layers.core import Dense, Reshape
from keras.layers import Input, Activation
from keras.layers import Concatenate
from keras.layers.merge import Dot
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model



from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy import sparse


from copyleaks.copyleakscloud import CopyleaksCloud
from copyleaks.product import Product
from copyleaks.processoptions import ProcessOptions
cloud = CopyleaksCloud(Product.Education, 'name@gmail.com', '######')# You can change the product.

stemmer = SnowballStemmer("english")
options = ProcessOptions()
options.setSandboxMode(True)

FOLDER_FIGURES = '../figures_definition'
FOLDER_MODELS = '../weights/server_trained'

CUSTOM_STOPWORDS = parsing.preprocessing.STOPWORDS.union(set(['privacy']))
# CUSTOM_STOPWORDS = parsing.preprocessing.STOPWORDS.union(set(['privacy' , 'inform', 'person', 'share', 'abil', 'know']))

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', 20)

logging.basicConfig(filename='custom_output.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in utils.simple_preprocess(text) :
        if token not in CUSTOM_STOPWORDS and len(token) > 3:
            result.append((token))
            # result.append(lemmatize_stemming(token))
            
    return result

def display_closestwords_tsnescatterplot(model, words_list, embedding=300, plot_name='skip_gram_test.png'):
    
    arr = np.empty((0,embedding), dtype='f')
    word_labels = []
    for label in words_list:
        word_labels.append([label, label])
    # word_labels = words_list.copy()

    # print('LENGTH: ', len(words_list))

    # get close words
    for word in words_list:
        # print('WORD: ', word)
        close_words = model.most_similar(word)
        print('WORD: ', word, ' close words: ', close_words)
        
        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word]]), axis=0)
        for wrd_score in close_words:
            # print('CLOSED_WORDS: ', wrd_score[0])
            wrd_vector = model[wrd_score[0]]
            word_labels.append([wrd_score[0], word])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
            
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for label, x, y in zip(word_labels, x_coords, y_coords):
        index_color = words_list.index((label[1]))
        if label[0] in words_list:
            plt.annotate(label[0], xy=(x, y), xytext=(0.5, 0.5), textcoords='offset points', weight='bold', color= colors[index_color])
           
        else:
            plt.annotate(label[0], xy=(x, y), xytext=(0.5, 0.5), textcoords='offset points', color= colors[index_color])
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.savefig(os.path.join(FOLDER_FIGURES, plot_name), bbox_inches='tight')
    plt.show()

# folder_survey_js = '../results/survey_js/amt_definition_total'
# filename_survey_js = 'privacy_survey_js_results_filtered_total.csv'
folder_survey_js = '../results/survey_js/amt_definition_total'
# filename_survey_js = 'privacy_survey_js_results_filtered_total.csv'
filename_survey_js = 'privacy_survey_js_results_filtered_total_frameworks_carlos.csv'

# folder_survey_js = '../results/survey_js/social_pilot'
# filename_survey_js = 'definitions_social_filtered_skipgram.csv'


likert_scale_concern = {'Very concerned' : 5,
                'Concerned' : 4,
                'Neither concerned nor unconcerned': 3,
                'Unconcerned' : 2,
                'Very unconcerned' : 1
                }
concern_bins = [0, 2, 4, 5]

definition_list = []
with open(os.path.join(folder_survey_js, filename_survey_js),encoding='utf-8') as csv_file:

    print(pd.get_option('display.encoding'))
    pd_survey = pd.read_csv(csv_file, delimiter=',')

    pd_survey.replace(likert_scale_concern, inplace= True)
    # print(pd_survey.head())

    # pd_survey['average_concern'] = pd_survey[['In general, how concerned are you about your privacy while you are using the Internet?',
    #                                         'Are you concerned that you are asked for too much personal information when you register or make online purchases?',
    #                                         'Are you concerned about people you do not know obtaining personal information about you from your online activities?',
    #                                         'Are you concerned that an email you send to someone may be read by someone else besides the person you sent it to?',
    #                                         'Are you concerned that an email you send to someone may be inappropriately forwarded to others? ',
    #                                         'Are you concerned who might access your medical records?']].mean(axis=1)


    # pd_survey['concern_bin'] = pd.cut(pd_survey['average_concern'], concern_bins, labels=['(0, 2]', '(2, 4]', '(4, 5]'])


    # pd_survey['familiarity_technology'] = np.select([(pd_survey['How often do you use a smartphone in a day? '] == '0 to 1 hour'),
    #                                         (pd_survey['How often do you use a smartphone in a day? '] == '1 to 2 hours'),
    #                                         (pd_survey['How often do you use a smartphone in a day? '] == '2 to 3 hours'),
    #                                         (pd_survey['How often do you use a smartphone in a day? '] == '3 to 4 hours'),
    #                                         (pd_survey['How often do you use a smartphone in a day? '] == 'more than 4 hours')], [1, 2, 3, 4, 5], default=0)


    
    # pd_survey['familiarity_smart_home'] = np.select([(pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'No')
    #                                         & (pd_survey['Are you planning to buy any smart device in the future?'] == 'No'),
    #                                         (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'No')
    #                                         & (pd_survey['Are you planning to buy any smart device in the future?'] == 'Yes'),
    #                                         (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'Yes')
    #                                         & (pd_survey['Are you planning to buy any smart device in the future?'] == 'No'),
    #                                         (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'Yes')
    #                                         & (pd_survey['Are you planning to buy any smart device in the future?'] == 'Yes')], [1, 3, 4, 5], default=0)
    

    # pd_survey['familiarity_computer_sec'] = np.select([(pd_survey['concern_bin'] == '(0, 2]'),
    #                                         (pd_survey['concern_bin'] == '(2, 4]'),
    #                                         (pd_survey['concern_bin'] == '(4, 5]')], [2, 4, 5], default =0)

    # pd_survey = pd_survey.dropna(subset=['Define privacy in your own words'])



    # gk_country = pd_survey.groupby('Country of residence')
    # gk_age = pd_survey.groupby('What is your age?')
    # gk_gender = pd_survey.groupby('What is your gender?')
    # gk_education = pd_survey.groupby('Education')
    # gk_job = pd_survey.groupby('What is your profession?')
    # gk_often_smartphone = pd_survey.groupby('How often do you use a smartphone in a day? ')
    # gk_do_you_have_iot = pd_survey.groupby('Do you have any smart device at home (e.g., smart tv, smart speaker)?')
    # gk_concern = pd_survey.groupby('concern_bin')

    # gk_familiarity_technology = pd_survey.groupby('familiarity_technology')
    # gk_familiarity_smart_home = pd_survey.groupby('familiarity_smart_home')
    # gk_familiarity_computer_security = pd_survey.groupby('familiarity_computer_sec')



    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Italy', 'Greece', 'Belgium',
    #                                             'France', 'Ireland', 'Spain', 'Netherlands', 'Denmark', 'Germany'
    #                                             'United Kingdom of Great Britain and Northern Ireland']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Brazil', 'Colombia', 'El Salvador', 'Guatemala',
    #                                             'Mexico', 'Venezuela (Bolivarian Republic of)']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Canada', 'United States of America']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Bangladesh', 'India']])['Define privacy in your own words']



    definition_text = (pd_survey['Define privacy in your own words'])


definition_text = [x for x in definition_text if str(x) != 'nan']

# print(definition_text)


# bible = (gutenberg.sents('bible-kjv.txt'))
# bible = bible[1:]

# gutenberg_text=[]
# for sentence in bible:
#     # print(sentence)
#     gutenberg_text.append(' '.join(i.lower() for i in sentence))

# print(gutenberg_text[0])

# definition_text = gutenberg_text

definition_text_lemmatized = []
definition_text_lemmatized_comma = []
for definition in definition_text:
    # print(definition)
    processed_text = preprocess(definition)
    definition_text_lemmatized.append(' '.join(i.lower() for i in processed_text))
    definition_text_lemmatized_comma.append([','.join(i.lower() for i in processed_text)])

print('Length definition text: ', len(definition_text_lemmatized))
print((definition_text_lemmatized[1000]))
print(definition_text[0])

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(definition_text_lemmatized)

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}



vocab_size = len(word2id) +1
embed_size = 300

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in definition_text_lemmatized]

print('Vocabulary Size: ', vocab_size)
print('Vocabulary Sample: ', list(word2id.items())[:10])

skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=5) for wid in wids]



pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(30):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
        id2word[pairs[i][0]], pairs[i][0], 
        id2word[pairs[i][1]], pairs[i][1], 
        labels[i]))

# print(definition_text_lemmatized_comma[:1])

# w2v = Word2Vec(definition_text_lemmatized_comma, size=embed_size, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
# word_vectors = w2v.wv
# result = word_vectors.similar_by_word('information')
# print("INFORMATION: ", result)
# exit(0)

# Load Model
# SkipGram = load_model(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_50.h5'))
# SkipGram = load_model(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_50_lemmatized.h5'))
# SkipGram = load_model(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_bible_50.h5'))
SkipGram = load_model(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_test_plot_400_windows_5.h5'))

# # Build skip-gram architecture
# word_inputs = Input(shape=(1, ), dtype='int32')
# word_model = Embedding(vocab_size, embed_size)(word_inputs)

# context_inputs = Input(shape=(1, ), dtype='int32')
# context_model = Embedding(vocab_size, embed_size)(context_inputs)

# output_model = Dot(axes=2)([word_model, context_model])
# output_model = Reshape((1, ), input_shape=(1, 1))(output_model)
# output_model = Activation('sigmoid')(output_model)

# SkipGram = Model(inputs=[word_inputs, context_inputs], outputs=output_model)
# print(SkipGram.summary())
# # logging.info(str(SkipGram.summary()))
# SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

# Plot Model
# SVG(model_to_dot(SkipGram, show_shapes=True, show_layer_names=False, rankdir='TB').create(prog='dot', format='svg'))




# # Train Model
# MAX_EPOCHS = 60
# counter = 0
# for _ in range(MAX_EPOCHS):
#     loss = 0.
#     for i, doc in enumerate(tokenizer.texts_to_sequences(definition_text_lemmatized)):
#         # print(doc)
#         data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=5, negative_samples=5.)
#         x = [np.array(x) for x in zip(*data)]
#         y = np.array(labels, dtype=np.int32)
#         if x:
#             loss += SkipGram.train_on_batch(x, y)

#     print("epoch: ", counter, "losses: ", loss)
#     logging.info("epoch: " + str(counter) + " losses: " + str(loss))
#     counter += 1

# # # Save model
# # SkipGram.save(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_50_lemmatized.h5'))
# # SkipGram.save(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_50.h5'))
# SkipGram.save(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_bible_50.h5'))
# SkipGram.save(os.path.join(FOLDER_MODELS, 'skip_gram_epochs_test_plot_60_new.h5'))


f = open(os.path.join(FOLDER_MODELS ,'vectors_5.txt') ,'w')
f.write('{} {}\n'.format(vocab_size-1, embed_size))
vectors = SkipGram.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()


w2v = KeyedVectors.load_word2vec_format(os.path.join(FOLDER_MODELS ,'vectors.txt'), binary=False)


# print('INFORMATION: ', w2v.most_similar(positive=['information']))
# print('PERSONAL: ', w2v.most_similar(positive=['personal']))
# print('ABILITY: ', w2v.most_similar(positive=['ability']))
# print('KNOWN: ', w2v.most_similar(positive=['known']))
# print('SHARED: ', w2v.most_similar(positive=['shared']))

display_closestwords_tsnescatterplot(w2v, words_list=['information', 'personal', 'share', 'able', 'ability', 'know'], plot_name='skip_gram_test.png')
# display_closestwords_tsnescatterplot(w2v, words_list=['information', 'personal', 'able', 'right', 'people'], plot_name='skip_gram_test.png')
# display_closestwords_tsnescatterplot(w2v, words_list=['inform', 'person', 'abil', 'share', 'know'])
# display_closestwords_tsnescatterplot(w2v, words_list=['personal'])
# display_closestwords_tsnescatterplot(w2v, words_list=['jesus', 'flood'])
# display_closestwords_tsnescatterplot(w2v, words_list= ['solitude', 'intimacy', 'reserve', 'anonymity'], plot_name='skip_gram_test.png')
# display_closestwords_tsnescatterplot(w2v, words_list= ['surveillance', 'identification'], plot_name='skip_gram_test.png')
# display_closestwords_tsnescatterplot(w2v, words_list= ['sender', 'receiver', ], plot_name='skip_gram_test.png')

# exit(0)

# merge_layer = SkipGram.layers[0]
# word_model = merge_layer.layers[0]
# word_embed_layer = word_model.layers[0]

weights = SkipGram.get_weights()[0]

distance_matrix = euclidean_distances(weights)
print(distance_matrix.shape)

weights_sparse = sparse.csr_matrix(weights)


cos_similarities = cosine_similarity(weights_sparse)


# words_to_check = ['information', 'personal', 'ability', 'known', 'shared']
# words_to_check = ['inform', 'person', 'abil', 'share', 'know']
# words_to_check= ['solitude', 'intimacy', 'reserve', 'anonymity']
words_to_check = ['information']
similar_words = {search_term: [id2word[idx] for idx in cos_similarities[word2id[search_term]-1].argsort()[1:4]+1] for search_term in words_to_check}
# similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:2]+1] for search_term in ['inform', 'person', 'abil', 'share', 'know']}

print('Similar words: ', similar_words)

exit(0)

words = sum([[k] + v for k, v in similar_words.items()], [])
words_ids = [word2id[w] for w in words]
word_vectors = np.array([weights[idx] for idx in words_ids])
print('Total words:', len(words), '\tWord Embedding shapes:', word_vectors.shape)
# print('Words vector', word_vectors)



tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_vectors)
labels = words
print(labels)


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

plt.figure(figsize=(14, 8))
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=20)

plt.scatter(T[:, 0], T[:, 1], c = 'steelblue', edgecolors='k', cmap='viridis')

index_color = 0
words_already_seen = []
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    if label in words_to_check: 
        # print(label)
        if (label in words_already_seen) == False:
            # print(label)
            index_color = words_to_check.index((label))
            plt.annotate(label, xy=(x, y), xytext=(0.5, 0.5), textcoords='offset points', weight='bold', color= colors[index_color], size=16)
            words_already_seen.append(label)
    else:
        plt.annotate(label, xy=(x, y), xytext=(0.5, 0.5), textcoords='offset points', color= colors[index_color], size=16)

plt.show()
plt.savefig(os.path.join(FOLDER_FIGURES, 'skip_gram_vectors_44_test.png'), bbox_inches='tight')