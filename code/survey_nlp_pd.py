import os
import pprint
import csv
import time
import pprint
import pandas as pd
import numpy as np
from scipy import stats

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib

from matplotlib import pyplot as plt
import seaborn as sns

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

from gensim import corpora, models, utils, parsing
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
from scipy.sparse import coo_matrix

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
# nltk.download('brown')
from nltk.tag.stanford import StanfordNERTagger

from nltk.corpus import stopwords

import pyLDAvis

import spacy
nlp = spacy.load("en_core_web_sm")

from copyleaks.copyleakscloud import CopyleaksCloud
from copyleaks.product import Product
from copyleaks.processoptions import ProcessOptions
cloud = CopyleaksCloud(Product.Education, 'solrac207@gmail.com', 'EDEF5CDE-2BC7-421F-A299-17D80B6D3245')# You can change the product.

stemmer = SnowballStemmer("english")
options = ProcessOptions()
options.setSandboxMode(True)

FOLDER_FIGURES = '../figures_definition'
MONGO_FOLDER = '../results/mongodb/amt_200_total'
# MONGO_FOLDER = '../results/mongodb/social_pilot'


device_mongodb_filename = 'data_devices.csv'
cookies_mongodb_filename = 'data_cookies.csv'


folder_survey_js = '../results/survey_js/amt_definition_total'
filename_survey_js = 'privacy_survey_js_results_filtered_total_frameworks_carlos.csv'

# folder_survey_js = '../results/survey_js/social_pilot'
# filename_survey_js = 'privacy_survey_js_results_filtered_total.csv'
# filename_survey_js = 'definitions_social_filtered.csv'

likert_scale_concern = {'Very concerned' : 5,
                'Concerned' : 4,
                'Neither concerned nor unconcerned': 3,
                'Unconcerned' : 2,
                'Very unconcerned' : 1
                }
# concern_bins = [0, 2, 4, 5]
# concern_bins = [0, 2, 3, 5]
concern_bins = [0, 3, 5]

definition_list = []

CUSTOM_STOPWORDS = parsing.preprocessing.STOPWORDS.union(set(['privacy', 'Privacy', 'shit']))
# CUSTOM_STOPWORDS = parsing.preprocessing.STOPWORDS.union(set(['privacy', 'shit', 'inform', 'person', 'share', 'abil', 'know']))
# CUSTOM_STOPWORDS = parsing.preprocessing.STOPWORDS.union(set(['privacy', 'Privacy', 'shit', 'nan', 'information', 'personal', 'share', 'ability', 'know']))  # no lemmatize

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 20)

pp = pprint.PrettyPrinter(indent=4)

# def autolabel(rects, axis_label):
#     for idx,rect in enumerate(axis_label):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 axis_label[idx],
#                 ha='center', va='bottom', rotation=90)
  	
def lemmatization_lda(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def findtags(tag_prefix, tagged_text, top_number=1):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(top_number)) for tag in cfd.conditions())

#Most frequently occuring words
def get_top_n_words(corpus, n=None, ngram_range=1):

    vec = CountVectorizer(ngram_range=(ngram_range, ngram_range), stop_words=CUSTOM_STOPWORDS, max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in utils.simple_preprocess(text) :
        if token not in CUSTOM_STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            # result.append((token))
            
    return result


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)




with open(os.path.join(folder_survey_js, filename_survey_js)) as csv_file:

    pd_definitions = pd.read_csv(os.path.join(folder_survey_js, filename_survey_js), delimiter=',')
    
    pd_survey = pd.read_csv(csv_file, delimiter=',')

    pd_survey.replace(likert_scale_concern, inplace= True)

    pd_survey=pd_survey.rename(columns = {'Random generated user ID':'userID'})

    # print(pd_survey.head())

    pd_survey['average_concern'] = pd_survey[['In general, how concerned are you about your privacy while you are using the Internet?',
                                            'Are you concerned that you are asked for too much personal information when you register or make online purchases?',
                                            'Are you concerned about people you do not know obtaining personal information about you from your online activities?',
                                            'Are you concerned that an email you send to someone may be read by someone else besides the person you sent it to?',
                                            'Are you concerned that an email you send to someone may be inappropriately forwarded to others?',
                                            'Are you concerned who might access your medical records?']].mean(axis=1)

    # print(pd_survey.head())
        
    # pd_survey['average_concern'] = pd_survey[[
    #                                      'Are you concerned that an email you send to someone may be inappropriately forwarded to others?',
    #                                         'Are you concerned who might access your medical records?'
    #                                         ]].mean(axis=1)



    print(pd_survey['average_concern'].describe())
    
    values_to_ci = pd_survey['average_concern'].dropna()
    # print((values_to_ci.mean(axis=0)))
    # print((values_to_ci.sem(axis=0)))

    ci = stats.t.interval(0.95, 1101, loc=(values_to_ci.mean(axis=0)),scale=values_to_ci.sem(axis=0))
    print(ci)
    # print(dd)


    # pd_survey['concern_bin'] = pd.cut(pd_survey['average_concern'], concern_bins, labels=['(0, 2]', '(2, 4]', '(4, 5]'])
    pd_survey['concern_bin'] = pd.cut(pd_survey['average_concern'], concern_bins, labels=['(0, 3]', '[4, 5]'])
    
    
    # gk_concern = pd_survey.groupby('concern_bin')
    # print(gk_concern.size())
  

    # print(pd_survey.head())
    # pd_survey['familiarity_technology'] = np.where(pd_survey['How often do you use a smartphone in a day? '] == '0 to 1 hour', 1, 0)
    # pd_survey['familiarity_technology'] = np.where(pd_survey['How often do you use a smartphone in a day? '] == '1 to 2 hours', 2, 0)
    # pd_survey['familiarity_technology'] = np.where(pd_survey['How often do you use a smartphone in a day? '] == '2 to 3 hours', 3, 0)
    # pd_survey['familiarity_technology'] = np.where(pd_survey['How often do you use a smartphone in a day? '] == '3 to 4 hours', 4, 0)
    # pd_survey['familiarity_technology'] = np.where(pd_survey['How often do you use a smartphone in a day? '] == 'more than 4 hours', 5, 0)

    pd_survey['familiarity_technology'] = np.select([(pd_survey['How often do you use a smartphone in a day? '] == '0 to 1 hour'),
                                            (pd_survey['How often do you use a smartphone in a day? '] == '1 to 2 hours'),
                                            (pd_survey['How often do you use a smartphone in a day? '] == '2 to 3 hours'),
                                            (pd_survey['How often do you use a smartphone in a day? '] == '3 to 4 hours'),
                                            (pd_survey['How often do you use a smartphone in a day? '] == 'more than 4 hours')], [1, 2, 3, 4, 5], default=1)

    # pd_survey['familiarity_smart_home'] = np.select([(pd_survey['Are you planning to buy any smart device in the future?'] == 'No')
    #                                         & (pd_survey['Are you planning to buy any smart device in the future?'] == 'No')], [1], default=0)
    # pd_survey['familiarity_smart_home'] = np.where(pd_survey['Are you planning to buy any smart device in the future?'] == 'Yes', 2, 0)
    # pd_survey['familiarity_smart_home'] = np.select([(pd_survey['Are you planning to buy any smart device in the future?'] == 'Yes')
                                            # & (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'No')], [3], default=0)
    # pd_survey['familiarity_smart_home'] = np.where(pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'Yes', 4, 0)
    
    pd_survey['familiarity_smart_home'] = np.select([(pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'No')
                                            & (pd_survey['Are you planning to buy any smart device in the future?'] == 'No'),
                                            (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'No')
                                            & (pd_survey['Are you planning to buy any smart device in the future?'] == 'Yes'),
                                            (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'Yes')
                                            & (pd_survey['Are you planning to buy any smart device in the future?'] == 'No'),
                                            (pd_survey['Do you have any smart device at home (e.g., smart tv, smart speaker)?'] == 'Yes')
                                            & (pd_survey['Are you planning to buy any smart device in the future?'] == 'Yes')], [1, 3, 4, 5], default=1)
   
    # pd_survey['familiarity_computer_sec'] = np.where(pd_survey['concern_bin'] == '(2, 4]', 4, 0)
    # pd_survey['familiarity_computer_sec'] = np.where(pd_survey['concern_bin'] == '(0, 2]', 2, 0)
    # pd_survey['familiarity_computer_sec'] = np.where(pd_survey['concern_bin'] == '(2, 4]', 4, 0)
    # pd_survey['familiarity_computer_sec'] = np.where(pd_survey['concern_bin'] == '(4, 5]', 5, 0)

    # pd_survey['familiarity_computer_sec'] = np.select([(pd_survey['concern_bin'] == '(0, 2]'),
    #                                         (pd_survey['concern_bin'] == '(2, 4]'),
    #                                         (pd_survey['concern_bin'] == '(4, 5]')], [2, 4, 5], default =0)

    pd_survey['familiarity_computer_sec'] = np.select([((pd_survey['concern_bin'] == '(0, 3]') 
                                                & (pd_survey['How often do you use a computer in a day?'] == '1 to 2 hour')),
                                            ((pd_survey['concern_bin'] == '(0, 3]') 
                                                & (pd_survey['How often do you use a computer in a day?'] == '3 to 4 hours' )),
                                            ((pd_survey['concern_bin'] == '[4, 5]')
                                                & (pd_survey['How often do you use a computer in a day?'] == 'more than 4 hours' ))
                                            ], [2, 3, 4], default =1)

    # print('familiarity_technology: ', pd_survey['familiarity_technology'].mean())
    # print('familiarity_smart_home: ', pd_survey['familiarity_smart_home'].mean())
    # print('familiarity_computer_sec: ', pd_survey['familiarity_computer_sec'].mean())

    definition_list= (pd_survey['Define privacy in your own words'])

    print(pd_survey.shape)

    pd_survey.to_csv('privacy_survey_js_results_curated.csv')

    # exit(0)

   


    with open(os.path.join(MONGO_FOLDER, device_mongodb_filename)) as csv_mongodb_file:
        pd_devices = pd.read_csv(csv_mongodb_file, delimiter=',')

        df_merge = pd.merge(pd_survey, pd_devices, on='userID')

        # print(df_merge.head())
    
        pd_survey = df_merge
        # print(pd_survey.head())
    
        # print(ddd)

    with open(os.path.join(MONGO_FOLDER, cookies_mongodb_filename)) as csv_cookies_file:
        pd_cookies = pd.read_csv(csv_cookies_file, delimiter=',')

        df_merge = pd.merge(pd_survey, pd_cookies, on='userID')

        
    
        pd_survey = df_merge
        

    pd_survey['defaultCookieSettings'] = pd_survey['essential'].map(str) + pd_survey['analytics'].map(str) + pd_survey['site_preferences'].map(str) + pd_survey['marketing'].map(str)
    pd_survey['userCookieSettings'] = pd_survey['user_essential'].map(str) + pd_survey['user_analytics'].map(str) + pd_survey['user_site_preferences'].map(str) + pd_survey['user_marketing'].map(str)

    pd_survey['userChangedCookieSettings'] = np.where(pd_survey['defaultCookieSettings'] == pd_survey['userCookieSettings'], False, True)

    pd_survey['acceptedCookie'] = np.where(pd_survey['time_to_accept'] == 0, False, True)

    pd_survey['screen_area'] = pd_survey['screen_width'] * pd_survey['screen_height']

    # print(pd_survey.head()

    # print(westin_df.head())
    # print(list(westin_df.columns.values))


    gk_country = pd_survey.groupby('Country of residence')
    gk_age = pd_survey.groupby('What is your age?')
    gk_gender = pd_survey.groupby('What is your gender?')
    gk_education = pd_survey.groupby('Education')
    gk_job = pd_survey.groupby('What is your profession?')
    gk_often_smartphone = pd_survey.groupby('How often do you use a smartphone in a day? ')
    gk_do_you_have_iot = pd_survey.groupby('Do you have any smart device at home (e.g., smart tv, smart speaker)?')
    gk_concern = pd_survey.groupby('concern_bin')

    gk_age_gender = pd_survey.groupby(['What is your age?', 'What is your gender?'])

    gk_familiarity_technology = pd_survey.groupby('familiarity_technology')
    gk_familiarity_smart_home = pd_survey.groupby('familiarity_smart_home')
    gk_familiarity_computer_security = pd_survey.groupby('familiarity_computer_sec')

    gk_browser = pd_survey.groupby('client_name')
    gk_device = pd_survey.groupby('device_class')
    gk_os = pd_survey.groupby('client_os')
    gk_firstTimeCookie = pd_survey.groupby('Have you ever interact with cookie consent notices (e.g., pop-up menu at the beginning) before this survey?')

    gk_cookieChanged = pd_survey.groupby('userChangedCookieSettings')
    gk_adBlocker = pd_survey.groupby('adblocker_enabled')
    gk_cookieChangedAndNudgeDefault = pd_survey.groupby(['userChangedCookieSettings', 'nudgeShowDefaultCookies'])
    gk_cookieChangedAndNudgeBar = pd_survey.groupby(['userChangedCookieSettings', 'nudgePrivacyBar'])
    gk_cookieChangedAndNudge = pd_survey.groupby(['userChangedCookieSettings', 'nudgeShowDefaultCookies', 'nudgePrivacyBar'])
    gk_acceptedCookie = pd_survey.groupby('acceptedCookie')

    gk_cookieChanged_bannerType = pd_survey.groupby(['userChangedCookieSettings', 'bannerType'])

    gk_os_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'client_os'])
    gk_device_changedCookie_adblocker = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled' ,'device_class'])

    gk_security_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'familiarity_computer_sec'])
    gk_adBlocker_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled'])
    gk_adBlocker_changedCookie_nudging = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled', 'nudgePrivacyBar'])

    gk_changedCookie_security = pd_survey.groupby(['userChangedCookieSettings', 'familiarity_computer_sec'])
    gk_adBlocker_changedCookie_security = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled', 'familiarity_computer_sec'])

    gk_bannerType_changedCookie_nudging = pd_survey.groupby(['userChangedCookieSettings', 'nudgeShowDefaultCookies', 'nudgePrivacyBar', 'bannerType'])

    gk_bannerType = pd_survey.groupby(['bannerType'])
    gk_nudgeDefaults = pd_survey.groupby(['nudgeShowDefaultCookies'])
    gk_nudgeProgressBar = pd_survey.groupby(['nudgePrivacyBar'])

    gk_age_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'What is your age?'])
    gk_country_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'Country of residence'])
    gk_concern_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'concern_bin'])
    gk_iot_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'Do you have any smart device at home (e.g., smart tv, smart speaker)?'])
    gk_adblocker_changedCookie_concern = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled' ,'concern_bin'])
    gk_adblocker_changedCookie_education = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled' ,'Education'])
    gk_adblocker_changedCookie_age = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled' ,'What is your age?'])

    gk_adblocker_changedCookie_gender = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled' ,'What is your gender?'])
    gk_adblocker_changedCookie_firstTimeCookie = pd_survey.groupby(['userChangedCookieSettings', 'adblocker_enabled' ,'Have you ever interact with cookie consent notices (e.g., pop-up menu at the beginning) before this survey?'])

    gk_client_name_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'client_name'])
    gk_wearable_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'Do you use any wearable (e.g., smart watch)?'])
    gk_time_computer_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'How often do you use a computer in a day?'])
  
    
    gk_age_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'What is your age?'])    

    gk_technology_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'familiarity_technology'])
    gk_smarthome_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'familiarity_smart_home'])
    gk_security_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'familiarity_computer_sec'])
    gk_gender_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'What is your gender?'])
    gk_firstTimeCookie_changedCookie = pd_survey.groupby(['userChangedCookieSettings', 'Have you ever interact with cookie consent notices (e.g., pop-up menu at the beginning) before this survey?'])


    gk_changedCookie_screenArea = pd_survey.groupby(['userChangedCookieSettings','screen_height'])
    # gk_changedCookie_screenArea = gk_changedCookie_screenArea.apply(lambda g: g[g['screen_height'] >= 20])

    # gk_cookieChangedAndNudgeAccepted = pd_survey.groupby(['acceptedCookie', 'userChangedCookieSettings', 'nudgeShowDefaultCookies', 'nudgePrivacyBar'])

    gk_adblocker_age = pd_survey.groupby(['adblocker_enabled' ,'What is your age?'])
    gk_adblocker_education = pd_survey.groupby(['adblocker_enabled' ,'Education'])
    gk_adblocker_country = pd_survey.groupby(['adblocker_enabled' ,'Country of residence'])
    gk_adblocker_concern = pd_survey.groupby(['adblocker_enabled' ,'concern_bin'])
    gk_adblocker_technology = pd_survey.groupby(['adblocker_enabled' ,'familiarity_technology'])
    gk_adblocker_smart_home = pd_survey.groupby(['adblocker_enabled' ,'familiarity_smart_home'])
    gk_adblocker_security = pd_survey.groupby(['adblocker_enabled' ,'familiarity_computer_sec'])

    gk_age_eudcation_os = pd_survey.groupby(['What is your age?', 'Education', 'client_os' ])

    gk_device_brand = pd_survey.groupby(['device_brand'])

    # print(gk_age.size())
    # print(gk_gender.size())
    print(gk_country.size())
    # print(gk_education.size())
    # print(gk_often_smartphone.size())
    # print(gk_do_you_have_iot.size())
    # print(gk_job.size())
    print(gk_concern.size())
    print(gk_familiarity_technology.size())
    print(gk_familiarity_computer_security.size())

    # print(pd_survey.head())
    # print(ddd)

    print(gk_device_brand.size())
    print(gk_age_changedCookie.size())
    # print(gk_age_gender.size())
    # print(dd)

    # print(gk_browser.size())
    # print(gk_device.size())
    # print(gk_os.size())
    # print(gk_os_changedCookie.size())

    # # Cookies changed
    # print(gk_cookieChanged.size())
    # print(gk_adBlocker.size())
    # print(gk_cookieChanged_bannerType.size())
    # print(gk_cookieChangedAndNudge.size())
    # print(gk_acceptedCookie.size())
    # print(gk_client_name_changedCookie.size())
    # print(gk_device_changedCookie_adblocker.size())

    # print(gk_security_changedCookie.size())
    # print(gk_adBlocker_changedCookie.size())
    # print(gk_adBlocker_changedCookie_nudging.size())
    # print(gk_firstTimeCookie.size())
    # print(gk_acceptedCookie.size())
    # print(gk_nudgeDefaults.size())
    # print(gk_nudgeProgressBar.size())
    # print(gk_browser.size())
    # print(gk_os.size())
    # print(gk_age_eudcation_os.size())

    # print(pd_survey.shape)

    # # Adblocker
    # print(gk_adblocker_age.size())
    # print(gk_adblocker_education.size())
    # print(gk_adblocker_country.size())
    # print(gk_adblocker_concern.size())
    # print(gk_adblocker_technology.size())
    # print(gk_adblocker_smart_home.size())
    # print(gk_adblocker_security.size())


    # print(gk_changedCookie_security.size())
    # print(gk_adBlocker_changedCookie_security.size())
    # print(gk_adblocker_changedCookie_concern.size())
    # print(gk_adblocker_changedCookie_education.size())
    # print(gk_adblocker_changedCookie_age.size())
    # print(gk_adblocker_changedCookie_firstTimeCookie.size())
    # print(gk_adblocker_changedCookie_gender.size())
    # print(gk_wearable_changedCookie.size())
    # print(gk_time_computer_changedCookie.size())
    # print(gk_firstTimeCookie_changedCookie.size())

    # print(gk_changedCookie_screenArea)

    print(gk_bannerType_changedCookie_nudging.size())
    print(gk_cookieChangedAndNudgeBar.size())
    print(gk_cookieChangedAndNudgeDefault.size())
    print(gk_cookieChangedAndNudge.size())

    # print(gk_age_changedCookie.size())
    # print(gk_country_changedCookie.size())
    # print(gk_concern_changedCookie.size())
    # print(gk_iot_changedCookie.size())
    # print(gk_technology_changedCookie.size())
    # print(gk_smarthome_changedCookie.size())
    # print(gk_security_changedCookie.size())
    # print(gk_gender_changedCookie.size())
    # print(dd)

    # print(gk_bannerType.size())

    # print(gk_westin.size())


    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [0, 1, 2, 3]])
    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [4, 5]])

    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [0,2]])
    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [5]])

    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_gender if name in ['Male']])

    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_concern if name in ['(0, 3]']])
    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_concern if name in ['[4, 5]']])


    # pd_survey_frameworks = pd.concat([group for (name, group) in gk_country if name in ['Italy', 'Greece', 'Belgldam',
    #                                             'France', 'Ireland', 'Spain', 'Netherlands', 'Denmark', 'Germany'
    #                                             'United Kingdom of Great Britain and Northern Ireland']])



    # pp = pd.concat([group for (name, group) in gk_os if name in ['macOS 10.12 Sierra', 'macOS 10.13 High Sierra', 'macOS 10.14 Mojave'
    #                                                                         'macOS 10.15 Catalina']])
    # pp = pd.concat([group for (name, group) in gk_os if name in ['Windows XP  ', 'Windwos Vista', 'Windows 8.1', 'Windows 8'
    #                                                                         'Windows 7', 'Windows 10']])
    # pp = gk_gender.get_group('Male')
    # pp = gk_cookieChanged.get_group(False)

    # pp =  pd.concat([group for (name, group) in gk_country if name in ['Italy', 'Greece', 'Belgium',
    #                                             'France', 'Ireland', 'Spain', 'Netherlands', 'Denmark', 'Germany',
    #                                             'United Kingdom of Great Britain and Northern Ireland']])

    # pp = gk_concern.get_group('(0, 3]')
    # pp = gk_concern.get_group('[4, 5]')

    # pp.to_csv('../../cookies_changed_false_subsample.csv')

    # print(dd)



    # # Extract NER information from the definitions
    # definition_text = (pd_survey['Define privacy in your own words'])
    # # definition_text = [x for x in definition_text if str(x) != 'nan']
    # print(definition_text)

    # doc = nlp(definition_text)

    # # Analyze syntax
    # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    # # Find named entities, phrases and concepts
    # for entity in doc.ents:
    #     print(entity.text, entity.label_)

    # exit(0)




    pd_survey_frameworks = pd_survey

    westin_df = pd_survey_frameworks["Westin"].str.split(",", n = -1, expand = True) 
    solove_df = pd_survey_frameworks["Solove"].str.split(",", n = -1, expand = True) 

    gk_westin = westin_df.apply(pd.value_counts)
    gk_westin = gk_westin.fillna(0)

    gk_solove = solove_df.apply(pd.value_counts)
    gk_solove = gk_solove.fillna(0)

    gk_westin['Freq'] = gk_westin.sum(axis = 1)
    gk_westin['category'] = gk_westin.index
    gk_westin['category'] = gk_westin['category'].str.strip()
    print(gk_westin.head())
    print(gk_westin.dtypes)

    number_labelled_participants = gk_westin['Freq'].sum()
    print('Number of labelled participants, Westin: ', number_labelled_participants)

    print('Category frequencies (number of participants): ', gk_westin.groupby('category')['Freq'].sum())
    print('Percentages:', gk_westin.groupby('category')['Freq'].sum()/number_labelled_participants)


    gk_westin_grouped = gk_westin.groupby('category')['Freq'].sum().to_frame()
    gk_westin_grouped['category'] = gk_westin_grouped.index
    gk_westin_grouped['Percentage'] = gk_westin_grouped['Freq']/number_labelled_participants * 100
    gk_westin_grouped['Framework'] = 'Westin'
    # print(gk_solove_grouped)

    gk_solove['Freq'] = gk_solove.sum(axis = 1)
    gk_solove['category'] = gk_solove.index
    gk_solove['category'] = gk_solove['category'].str.strip()
    

    number_labelled_participants = gk_solove['Freq'].sum()
    print('Number of labelled participants, Solove: ', number_labelled_participants)

    print('Category frequencies (number of participants): ', gk_solove.groupby('category')['Freq'].sum())
    print('Percentages:', gk_solove.groupby('category')['Freq'].sum()/number_labelled_participants)

    gk_solove_grouped = gk_solove.groupby('category')['Freq'].sum().to_frame()
    gk_solove_grouped['category'] = gk_solove_grouped.index
    gk_solove_grouped['Percentage'] = gk_solove_grouped['Freq']/number_labelled_participants * 100
    gk_solove_grouped['Framework'] = 'Solove'
    # print(gk_solove_grouped)
  

    df_privacy_frameworks = pd.concat([gk_westin_grouped, gk_solove_grouped])
    print(df_privacy_frameworks)
    # print(d)

    x_axis_label = df_privacy_frameworks['category']

    f, axes = plt.subplots(sharey=True, figsize=(36, 20))
    # bp = sns.boxplot(x='category', y='Freq', 
    #                 hue='category', data=gk_westin, palette="Set2")
    fc = sns.barplot(x="category", y="Percentage", errwidth=14, capsize=.2, hue="Framework", data=df_privacy_frameworks, palette='pastel')
    # fc = sns.factorplot(x='category', y="Freq", hue="category", data=gk_solove_grouped, kind="bar", 
    #                 palette="Set1")
    # plt.show()
    # autolabel(fc, x_axis_label)
    # ax.plot([1],[1])
    # axes.tick_params(axis=u'both', which=u'both',length=0)
    # find the values and append to list
    # totals = []
    # for i in fc.patches:
    #     totals.append(i.get_height())

    # # set individual bar lables using above list
    # total = sum(totals)
    # set individual bar lables using above list

    hatches = ['/', '+', '-', 'x', '\\', '*', 'o', 'O', '.']
    current_palette = sns.color_palette('pastel')
    index_label = 0
    index_hatch = 0
    for i, bar in enumerate(fc.patches):
        if np.isnan(bar.get_height()) == False:
        # get_x pulls left or right; get_height pushes up or down
            fc.text(bar.get_x()+.0, -1.0 + len(x_axis_label[index_label]), str(x_axis_label[index_label]), fontsize=100, color='black', rotation=90)
            bar.set_hatch(hatches[index_hatch])
            index_label += 1
            if index_label % 4 == 0:
                index_hatch += 1
    patch_1 = matplotlib.patches.Patch(facecolor=current_palette[0], label='Westin', hatch=hatches[0])
    patch_2 = matplotlib.patches.Patch(facecolor=current_palette[1], label='Solove', hatch=hatches[1])

    # add legends
    plt.legend(handles=[patch_1, patch_2], loc='best', fontsize=70)

    # axes.legend(fontsize=70)
    plt.xticks(rotation=60, fontsize=100)
    plt.yticks(fontsize=100)
    fc.set(xticklabels=[])
    plt.xlabel('Codes', fontsize=100,  weight='bold')
    # plt.ylabel('Percentage', fontsize=100)
    plt.ylabel('Percentage (%)', fontsize=100,  weight='bold')
    sns.set(font_scale = 2)
    # sns.set_style("ticks")
    # sns.set_style("ticks", {"ytick.major.size": 100})
    axes.tick_params(axis='y', length=20, width=10)
    plt.savefig(os.path.join(FOLDER_FIGURES, 'privacy_frameworks_all_barplot.png'), bbox_inches='tight')

    plt.close()

    # exit(0)

    # print(dd)

    # cangedCookie_users = pd_survey.loc[pd_survey['userChangedCookieSettings'] ==  True]
    # print(cangedCookie_users['What is your profession?'])

    # collegeStudents = pd_survey.loc[pd_survey['Education'] ==  'Some college but no degree']
    # definition_text = (collegeStudents['Define privacy in your own words'])

    # print(gk_cookieChangedAndNudgeAccepted.size())

    # print(gk_age.agg({'What is your age?':'sum'}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())))


    # definition_text = gk_gender.get_group('Female')['Define privacy in your own words']
    # definition_text = gk_gender.get_group('Male')['Define privacy in your own words']

    # print(gk_age.get_group('25 to 34'))
    # Privacy definition
    # definition_text = gk_country.get_group('United Kingdom of Great Britain and Northern Ireland')['Define privacy in your own words']
    # definition_text = gk_country.get_group('Brazil')['Define privacy in your own words']
    # definition_text = gk_country.get_group('Canada')['Define privacy in your own words']
    # definition_text = gk_country.get_group('India')['Define privacy in your own words']
    # definition_text = gk_country.get_group('United States of America')['Define privacy in your own words']
    # definition_text = gk_country.get_group('Italy')['Define privacy in your own words']

    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Italy', 'Greece', 'Belgium',
    #                                             'France', 'Ireland', 'Spain', 'Netherlands', 'Denmark', 'Germany'
    #                                             'United Kingdom of Great Britain and Northern Ireland']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Brazil', 'Colombia', 'El Salvador', 'Guatemala',
    #                                             'Mexico', 'Venezuela (Bolivarian Republic of)']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Canada', 'United States of America']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_country if name in ['Bangladesh', 'India']])['Define privacy in your own words']
    
    # definition_text = gk_age.get_group('18 to 24')['Define privacy in your own words']
    # definition_text = gk_age.get_group('25 to 34')['Define privacy in your own words']
    # definition_text = gk_age.get_group('35 to 44')['Define privacy in your own words']
    # definition_text = gk_age.get_group('45 to 54')['Define privacy in your own words']
    # definition_text = gk_age.get_group('55 to 64')['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_age if name in ['55 to 64', '65 to 74']])['Define privacy in your own words']

    # definition_text = gk_often_smartphone.get_group('0 to 1 hour')['Define privacy in your own words']
    # definition_text = gk_often_smartphone.get_group('1 to 2 hours')['Define privacy in your own words']
    # definition_text = gk_often_smartphone.get_group('2 to 3 hours')['Define privacy in your own words']
    # definition_text = gk_often_smartphone.get_group('3 to 4 hours')['Define privacy in your own words']
    # definition_text = gk_often_smartphone.get_group('more than 4 hours')['Define privacy in your own words']


    # definition_text = gk_education.get_group('Associate degree')['Define privacy in your own words']
    # definition_text = gk_education.get_group('Bachelor degree')['Define privacy in your own words']
    # definition_text = gk_education.get_group('Graduate degree')['Define privacy in your own words']
    # definition_text = gk_education.get_group('High school degree or equivalent (e.g., GED)')['Define privacy in your own words']
    # definition_text = gk_education.get_group('Postgraduate degree')['Define privacy in your own words']
    # definition_text = gk_education.get_group('Some college but no degree')['Define privacy in your own words']

    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Teacher', 'Teachers Assistant']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Technical', 'Technical Advisor', 'Technical Analyst',
    #                                                     'Technical Clerk', 'Technical Co-ordinator', 'Technical Engineer',
    #                                                     'Technical Manager']])['Define privacy in your own words']

    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Administration Assistant', 'Administration Clerk',
    #                                                     'Administration Manager', 'Administration Staff',
    #                                                     'Administratior']])['Define privacy in your own words']       
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Accountant', 'Accounts Assistant',
    #                                                     'Accounts Clerk', 'Accounts Manager',
    #                                                     'Accounts Staff']])['Define privacy in your own words']      
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Analyst']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Architect']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Consultant']])['Define privacy in your own words']      
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Doctor']])['Define privacy in your own words']                                                
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Engineer']])['Define privacy in your own words']
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['IT Consultant', 'IT Manager', 'IT Trainer']])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Office Manager', 'Office Worker']])['Define privacy in your own words']   
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Research Analyst', 'Research Consultant', 
    #                                                     'Research Scientist', 'Researcher', 'Scientist']])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Retired']])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Sales Administration', 'sales Executive',
    #                                                     'Sales Manager', 'Sales Representative',
    #                                                     'Salesman', 'Saleswoman']])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Software Consultant', 'Software Engineer']])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_job if name in ['Student']])['Define privacy in your own words'] 

    # definition_text = gk_concern.get_group('(0, 3]')['Define privacy in your own words']
    # definition_text = gk_concern.get_group('[4, 5]')['Define privacy in your own words']

    # definition_text = gk_concern.get_group('(0, 2]')['Define privacy in your own words']
    # definition_text = gk_concern.get_group('(2, 4]')['Define privacy in your own words']
    # definition_text = gk_concern.get_group('(4, 5]')['Define privacy in your own words']


    # definition_text = gk_do_you_have_iot.get_group('Yes')['Define privacy in your own words']
    # definition_text = gk_do_you_have_iot.get_group('No')['Define privacy in your own words']

    # definition_text = pd.concat([group for (name, group) in gk_familiarity_technology if name in [0, 1, 2, 3]])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_familiarity_technology if name in [3]])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_familiarity_technology if name in [4, 5]])['Define privacy in your own words'] 

    # definition_text = pd.concat([group for (name, group) in gk_familiarity_smart_home if name in [0, 1, 2, 3]])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_familiarity_smart_home if name in [3]])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_familiarity_smart_home if name in [4, 5]])['Define privacy in your own words'] 

    # definition_text = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [0, 1, 2]])['Define privacy in your own words'] 
    # definition_text = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [3]])['Define privacy in your own words'] $ NO DATA
    # definition_text = pd.concat([group for (name, group) in gk_familiarity_computer_security if name in [4, 5]])['Define privacy in your own words'] 


    # definition_text = gk_cookieChanged.get_group(True)['Define privacy in your own words']
    # definition_text = gk_cookieChanged.get_group(False)['Define privacy in your own words']


    # definition_text = pd_survey[(pd_survey['userChangedCookieSettings'] == True) & (pd_survey['adblocker_enabled'] == 1) &
    #                      (pd_survey['nudgePrivacyBar'] == 1)]['Define privacy in your own words'] 
    # definition_text = pd_survey[(pd_survey['userChangedCookieSettings'] == True) & (pd_survey['adblocker_enabled'] == 1) &
    #                      (pd_survey['nudgePrivacyBar'] == 0)]['Define privacy in your own words'] 
    # definition_text = pd_survey[(pd_survey['userChangedCookieSettings'] == False)]['Define privacy in your own words'] 
    # definition_text = pd_survey[(pd_survey['userChangedCookieSettings'] == True)]['Define privacy in your own words'] 

    # print(len(definition_text))
    # print(definition_text)

    definition_text = pd.concat([group for (name, group) in gk_browser if name in ['Chrome Mobile', 'Chrome'
                                                                                    ]])['Define privacy in your own words'] # Chrome users

    # definition_text = pd.concat([group for (name, group) in gk_browser if name in ['IE', 'Microsoft Edge'
    #                                                                             ]])['Define privacy in your own words'] # IE users

     # definition_text = pd.concat([group for (name, group) in gk_browser if name in ['Safari mobile', 'Safari'
    #                                                                             ]])['Define privacy in your own words'] # Safari users

    # definition_text = pd.concat([group for (name, group) in gk_browser if name in ['Firefox', 'Firefox mobile', 'Firefox for iOS', ,'Firefox'
    #                                                                                 ]])['Define privacy in your own words'] # Firefox users

    # definition_text = gk_device.get_group('Desktop')['Define privacy in your own words']

    # definition_text = pd.concat([group for (name, group) in gk_device if name in ['Smartphone', 'Tablet'
    #                                                                                 ]])['Define privacy in your own words'] # Smart mobile users

    # definition_text = pd.concat([group for (name, group) in gk_os if name in ['Windows XP  ', 'Windwos Vista', 'Windows 8.1', 'Windows 8'
    #                                                                         'Windows 7', 'Windows 10']])['Define privacy in your own words'] # Windows users
    # definition_text = pd.concat([group for (name, group) in gk_os if name in ['macOS 10.12 Sierra', 'macOS 10.13 High Sierra', 'macOS 10.14 Mojave'
    #                                                                         'macOS 10.15 Catalina']])['Define privacy in your own words'] # macOS users
    # definition_text = pd.concat([group for (name, group) in gk_os if name in ['Android 9.0 Pie', 'Android 8.1 Oreo', 'Android 7.1 Nougat', 'Android 7.0 Nougat'
    #                                                                         'Android 6.0 Marshmallow', 'Android 4.4 KitKat', 'Android 10.0'
    #                                                                         ]])['Define privacy in your own words'] # Android users
    # definition_text = pd.concat([group for (name, group) in gk_os if name in ['iOS 11', 'iOS 12', 'iOS 13', 'iPadOS'
    #                                                                         ]])['Define privacy in your own words'] # iOS users

    # definition_text = np.where(pd_survey['What is your age?']== '18 to 24' & pd_survey[] ==)


    # People who has no hope
    # userID_list = ['3c63b9fd-145a-40d5-bba1-811ad336efc6','84400307-b74e-495e-a850-9623761ad44a', 'b9d2b1c5-9a86-4e00-aaba-fec784a572c1', '9ac285b6-58cc-4894-b283-3d4718b6610d']

    # gk_no_hope = pd_survey[pd_survey['userID'].isin(userID_list)]
    # print(gk_no_hope)
    # definition_text = gk_no_hope['Define privacy in your own words']


    # definition_text = (pd_survey['Define privacy in your own words'])

    print(definition_text.head(10))

    print('Size definition_text: ', definition_text.size)



    # definition_text = pd_definitions['Definition']

    # print(definition_text)
    # process = cloud.createByText(definition_text)
    # print ("Submitted. In progress...")
    # iscompleted = False
    # while not iscompleted:
    #     # Get process status
    #     [iscompleted, percents] = process.isCompleted()
    #     print ('%s%s%s%%' % ('#' * int(percents / 2), "-" * (50 - int(percents / 2)), percents))
    #     if not iscompleted:
    #         time.sleep(2)
    # results = process.getResults()
    # print ('\nFound %s results...' % (len(results)))
    # print ("Process Finished!")
    # for result in results:
    #     print('')
    #     print('------------------------------------------------')
    #     print(result)

    # # Get the scan results
    # if len(results) == 0:



definition_text = [x for x in definition_text if str(x) != 'nan']

count = pd_survey['Define privacy in your own words'].str.split().str.len()
# count.index = count.index.astype(str) + ' words:'
count.sort_index(inplace=True)
print("Average number of words: ", np.mean(count))
print("Max number of words: ", np.max(count))
print("Min number of words: ", np.min(count))

# print('Number of definitions greater than 20: ', (np.nonzero(count>16)))
# print(d)
        

list_of_list_of_tokens = []

# definition_text = pd_survey['Define privacy in your own words']

for defintion_item in definition_text:
 
    # Privacy definition
    text_to_tokenizer = defintion_item
    # print(text_to_tokenizer)
    if isinstance(text_to_tokenizer, str):
        if len(text_to_tokenizer) != 0:
            tokenized_text = preprocess(text_to_tokenizer)
            list_of_list_of_tokens.append(tokenized_text)

    # Describe your answer (26 + 2; 16 describe in total, for IoT and web)
    # print(row[28:(28+15)])
    # for text_to_tokenizer in row[28:(28+15)]:
    #     # print(text_to_tokenizer)
    #     tokenized_text = preprocess(text_to_tokenizer)
    #     list_of_list_of_tokens.append(tokenized_text)

# print('list_of_list_of_tokens : ',list_of_list_of_tokens)

# wordfreq = []
# for w in list_of_list_of_tokens:
#     wordfreq.append(wordlist.count)

# print(list_of_list_of_tokens)

dictionary = corpora.Dictionary(list_of_list_of_tokens)
count = 0
num_topics = 20
for k, v in dictionary.iteritems():
    # print(k, v)
    count += 1
    if count > num_topics:
        break


dictionary.filter_extremes(no_below=5, no_above=0.1, keep_n= 100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in list_of_list_of_tokens]
document_num = 10
bow_doc_x = bow_corpus[document_num]

# for i in range(len(bow_doc_x)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
#                                                      dictionary[bow_doc_x[i][0]], 
#                                                      bow_doc_x[i][1]))



# lda_model =  models.LdaMulticore(bow_corpus, 
#                                    num_topics = 3, 
#                                    id2word = dictionary,                     
#                                    passes = 10,
#                                    workers = 2)

# for idx, topic in lda_model.print_topics(-1):
#     print("Topic: {} \nWords: {}".format(idx, topic ))
#     print("\n")

# print(list_of_list_of_tokens)

bigram = models.Phrases(list_of_list_of_tokens, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = models.phrases.Phraser(bigram)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'privacy', 'Privacy', 'shit'])

def remove_stopwords(texts):
    return [[word for word in models.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

data_words_nostops = remove_stopwords(list_of_list_of_tokens)
data_words_bigrams = make_bigrams(data_words_nostops)
# nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization_lda(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# print(data_lemmatized[:1])

dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)

dictionary_LDA = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
# dictionary_LDA.filter_extremes(no_below=3,no_above=0.1, keep_n= 100000)
corpus = [dictionary_LDA.doc2bow(text) for text in texts]

## Extract the topics - Thematic coding

# num_topics = 26
# # num_topics = 14
# # lda_model = models.LdaModel(corpus, num_topics=num_topics, \
# #                                 id2word=dictionary_LDA, \
# #                                 passes=40, alpha=[0.01]*num_topics, \
# #                                 eta=[0.01]*len(dictionary_LDA.keys()),
# #                                 per_word_topics=True)

# # for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics,num_words=10):
# #     print(str(i)+": "+ topic)
# #     print()


# # # Compute Perplexity
# # print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# # # Compute Coherence Score
# # coherence_model_lda = models.CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary_LDA, coherence='c_v')
# # coherence_lda = coherence_model_lda.get_coherence()
# # print('\nCoherence Score: ', coherence_lda)


# # mallet_path = './mallet-2.0.8/bin/mallet' # update this path
# # ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary_LDA)

# # # Show Topics
# # pp.pprint(ldamallet.show_topics(formatted=False))

# # # Compute Coherence Score
# # coherence_model_ldamallet = models.CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=dictionary_LDA, coherence='c_v')
# # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# # print('\nCoherence Score: ', coherence_ldamallet)



# # Can take a long time to run.
# # model_list, coherence_values = compute_coherence_values(dictionary=dictionary_LDA, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

# # # Show graph
# # limit=40
# # start=2
# # step=6

# # plt.rcParams.update(plt.rcParamsDefault)
# # plt.clf()
# # x = range(start, limit, step)
# # plt.plot(x, coherence_values)
# # plt.xlabel("Num Topics")
# # plt.ylabel("Coherence score")
# # plt.legend(("coherence_values"), loc='best')
# # plt.show()

# # # Print the coherence scores
# # for m, cv in zip(x, coherence_values):
# #     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# # Select the model and print the topics
# # optimal_model = model_list[4]
# optimal_model = ldamallet
# model_topics = optimal_model.show_topics(formatted=False)
# pp.pprint(optimal_model.print_topics(num_words=10))

# # exit(0) 

# df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=definition_text)

# # Format
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# # Show
# print('Dominant topic for each document/definition')
# print(df_dominant_topic.head(10))

# # Group top 5 sentences under each topic
# sent_topics_sorteddf_mallet = pd.DataFrame()

# sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

# for i, grp in sent_topics_outdf_grpd:
#     sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
#                                              grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
#                                             axis=0)

# # Reset Index    
# sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# # Format
# sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# # Show
# print('most representative document for each topic')
# print(sent_topics_sorteddf_mallet.head())

# sent_topics_sorteddf_mallet.to_csv('./most_representative_definitions_per_topic.csv')


# df_dominant_topic = df_dominant_topic.rename(columns={"Text": "Define privacy in your own words"})
# print(df_dominant_topic.head())
# print(pd_survey.head(2))
# df_merged = pd.merge(pd_survey, df_dominant_topic, on='Define privacy in your own words')

# print(df_merged.head())

# df_merged.to_csv('./merged_labelled_survey_14.csv')

# # exit(0)

# # Number of Documents for Each Topic
# topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# print('Number of documents per topic')
# print(topic_counts)

# # Percentage of Documents for Each Topic
# topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# # Topic Number and Keywords
# topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# # Concatenate Column wise
# df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# # Change Column names
# df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# # Show
# print('topic distribution accross documents')
# print(df_dominant_topics.head(30))

# df_dominant_topics.to_csv('./dominant_topic_df_14.csv')

# exit(0)

## Tf-IDF for each topic
# open csv 
# subset_pd_survey =  pd.read_csv('../results/survey_js/amt_definition_total/resuts_14_topics/merged_labelled_survey_14.csv', delimiter=',')
# print(subset_pd_survey.head())
# subset_pd_survey = subset_pd_survey.loc[subset_pd_survey['Dominant_Topic'] == 0.0]
# print(list_of_list_of_tokens.head())
# list_of_list_of_tokens = subset_pd_survey['Define privacy in your own words']

print(list_of_list_of_tokens[0])

# print(list_of_list_of_tokens)
corpus_text = []
for item in list_of_list_of_tokens:
    corpus_text.append(' '.join(i.lower() for i in item))

# print(corpus_text[0])
cv=CountVectorizer(max_df=0.8,stop_words=CUSTOM_STOPWORDS, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus_text)

# print(list(cv.vocabulary_.keys())[:10])

number_participants = len(definition_text)
print('LENGTH: ', number_participants)

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus_text, n=20, ngram_range=1)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
top_df['Percentage'] = top_df['Freq'] / number_participants * 100

# max_value = top_df['Freq'].max()
# print(max_value)
# y_tick_labels = np.arange(0, max_value,   step=20)
# print(y_tick_labels)



#Barplot of most freq words
f, axes = plt.subplots(sharey=True, figsize=(36, 20))
plt.xticks(rotation=70, ha='right', fontsize=100)
plt.yticks(fontsize=100)
sns.set(font_scale = 2)
# bp = sns.boxplot(x='Word', y='Freq', 
#                 hue='category', data=gk_westin, palette="Set2")
fc = sns.barplot(x="Word", y="Percentage", errwidth=14, capsize=.2, data=top_df)
# plt.show()
plt.xlabel('Word', fontsize=100)
plt.ylabel('Percentage [%]', fontsize=100)
plt.savefig(os.path.join(FOLDER_FIGURES, 'total_top_words_percent.png'), bbox_inches='tight')
# print(dd)


# f, axes = plt.subplots(sharey=True, figsize=(26, 40))
# # sns.set_style("whitegrid")
# # sns.set_style('white', {'axes.linewidth': 0.5})
# # plt.tight_layout()
# # sns.set(rc={'figure.figsize':(60,82)})
# # sns.set(style="white")
# # sns.set(font_scale = 20)
# # plt.rcParams.update({'font.size': 80})
# plt.xticks(rotation=70, fontsize=100, ha='right')
# plt.yticks(fontsize=100)
# plt.rcParams['xtick.major.size'] = 10
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['xtick.bottom'] = True
# plt.rcParams['ytick.left'] = True
# g = sns.barplot(x="Word", y="Freq", errwidth=14, capsize=.2, data=top_df)
# # g.set_xticklabels(g.get_xticklabels(), rotation=60, fontsize=80)
# # g.set_yticklabels(y_tick_labels, fontsize=80)
# g.set_ylabel("Freq",fontsize=100)
# g.set_xlabel("Words",fontsize=100)
# # fig = g.get_figure()
# # plt.show()
# plt.savefig(os.path.join(FOLDER_FIGURES, 'familiarity_computer_security_4_5_barplot_no_lem.png'), bbox_inches='tight')



tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)# get feature names
feature_names=cv.get_feature_names()

 
# fetch document for which keywords needs to be extracted
doc=""
for item in corpus_text:
    # print(item)
    doc = doc + " " + item
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items, 20)

keywords_100 = extract_topn_from_vector(feature_names,sorted_items, 100)

print(keywords_100)
# exit(0)
 
# now print the results
print("\nKeywords:")
tfidf_df = pd.DataFrame(columns = ['Word', 'Percentage'])
for k in keywords:
    print(k,keywords[k])
    doc = nlp(k)
    if k != 'person inform':
        tfidf_df = tfidf_df.append({'Word': k, 'Percentage' : (float(keywords[k])*100) }, ignore_index=True)
    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)

print(tfidf_df.head())

# tfidf_df = pd.DataFrame.from_dict(keywords)

#Barplot of most freq words
f, axes = plt.subplots(sharey=True, figsize=(36, 20))
plt.xticks(rotation=70, fontsize=100)
plt.yticks(fontsize=100)
sns.set(font_scale = 2)
# bp = sns.boxplot(x='Word', y='Freq', 
#                 hue='category', data=gk_westin, palette="Set2")
fc = sns.barplot(x="Word", y="Percentage", errwidth=14, capsize=.2, data=tfidf_df)
# plt.show()
plt.xlabel('Word', fontsize=100,  weight='bold')
plt.ylabel('Percentage (%)', fontsize=100, weight='bold')
plt.savefig(os.path.join(FOLDER_FIGURES, 'total_top_words_percent_tfidf.png'), bbox_inches='tight')

# exit(0)


top_words_lemmatize_list = []
for k in keywords:
    top_words_lemmatize_list.append(k)

# top_words_lemmatize_list = ['solitude', 'intimacy', 'reserve', 'anonymity']

# top_words_lemmatize_list = []
# for top_word in top_words_list:
#     tokenized_text = preprocess(top_word)
#     top_words_lemmatize_list = tokenized_text
# print(top_words_lemmatize_list)

# Get the frequency of participants using the top words
freq_participants_dict = {}
for  index, definition in enumerate(definition_text):
    # print(definition)
    if pd.isna(definition) == False:
        tokenized_text = preprocess(definition)
        # print(tokenized_text)
        for word in top_words_lemmatize_list:
            # print(word)
            if word in tokenized_text:
                if word in freq_participants_dict:
                    freq_participants_dict[word] += 1
                else:
                    freq_participants_dict[word] = 1
                tokenized_text = list(filter(lambda a: a != word, tokenized_text))
print('Freq participants: ', freq_participants_dict)
print('Total participants:', len(definition_text))        

# exit(0)

# documents = [open(f) for f in text_files]
# tfidf = TfidfVectorizer().fit_transform(documents)
# # no need to normalize, since Vectorizer will return normalized tf-idf
# similarity_matrix = cosine_similarity(tfidf)
# pairwise_similarity = tfidf * tfidf.T





# print(definition_text[0])

search_keywords = ['inform', 'person', 'share', 'abil', 'know']
# search_keywords = ['inform', 'person', 'share', 'abil', 'know', 'want', 'peopl', 'right', 'thing', 'privat']
# search_keywords = ['personal', 'able', 'other', 'own', 'private'] #adjectives


min_keywords = 4
definition_top_keywords = []
keyword_definition_df = pd.DataFrame(columns=['Text', 'Num_keywords'])

# offset_index = 12
# for  index, definition in enumerate(list_of_list_of_tokens):
for  index, definition in enumerate(definition_text):
    # print(definition)
    if pd.isna(definition) == False:
        definition_lemmatize = preprocess(definition)
        # print(definition_lemmatize)
        number_keywords = sum(1 for word in search_keywords if word in definition_lemmatize)
        keyword_definition_df = keyword_definition_df.append({'Text': definition, 'Num_keywords': number_keywords}, ignore_index=True)
        if number_keywords >= min_keywords:
            print('number of keywords ', number_keywords,'; index: ', index, '. ',definition)
            definition_top_keywords.append([definition])
            #    print(definition_text[index + offset_index])

keyword_definition_df = keyword_definition_df.rename(columns={"Text": "Define privacy in your own words"})
print(keyword_definition_df.head())
# print(pd_survey.head(2))
df_merged_keyword = pd.merge(pd_survey, keyword_definition_df, on='Define privacy in your own words')

df_merged_keyword.to_csv('./keywords_label.csv')

# exit(0)

# definition_text = gk_age.get_group('18 to 24')['Define privacy in your own words']


# plot wordcloud
# print(definition_text[29])
# text = definition_text
text= " ".join(str(review) for review in definition_text)
# print(text)

# Create and generate a word cloud image:
# wordcloud = WordCloud().generate(text)
# stopwords = set(STOPWORDS)
# stopwords.update(["privacy", "nan", "cookies", "shit"])

wordcloud = WordCloud(stopwords=CUSTOM_STOPWORDS, collocations=False, max_words=40, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.show()

wordcloud.to_file(os.path.join(FOLDER_FIGURES, 'wordcloud_familiarity_security_4_5_no_top_words.png'))

# print(dd)

print('Top lemmatized list: ', top_words_lemmatize_list)

# Find adjectives and adverbs
tag_list = []
for definition in top_words_lemmatize_list:
    # print(definition)
    if pd.isna(definition) == False:
        text = definition
        # Remove the top-5 most common used words
        # for stop_word in CUSTOM_STOPWORDS:
        #     if stop_word in text:
        #         text = list(filter(lambda a: a != stop_word, text))
        tags = nltk.pos_tag(text)
        print(tags)
        # print(tags[0])
        tag_list = tag_list + tags

exit(0)



# Extract NER information from the definitions
keywords_text = ' '.join(str(key) for key in keywords)
# print(keywords_text)

definition_top_keywords = ''.join(str(key) for key in definition_top_keywords)
print(definition_top_keywords)
doc = nlp(definition_top_keywords)
# doc = nlp('Privacy is the ability to hold, or share personal information as I see fit, not as an organization or government has deemed appropriate.')
pp.pprint([(X.text, X.label_) for X in doc.ents])

# exit(0)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)


print(tag_list[0])
tag_list_total = ",".join(str(review) for review in tag_list)
print('tag list: ', tag_list_total)


# tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
# for tag in sorted(tagdict):
#     print(tag, tagdict[tag])


tagdict = findtags('NN', tag_list, 10)


for tag in sorted(tagdict):
    print(tag, tagdict[tag])

adj_list = tagdict['NN']
words = [str(x[0]) for x in adj_list]
words_frequency = [x[1] for x in adj_list]
# print(words_frequency)

pd_adjs = pd.DataFrame({'Nouns': words, 'Freq': words_frequency})


tagdict = findtags('JJ', tag_list, 10)

for tag in sorted(tagdict):
    print(tag, tagdict[tag])

adj_list = tagdict['JJ']
words = [str(x[0]) for x in adj_list]
words_frequency = [x[1] for x in adj_list]
# print(words_frequency)

pd_adjs = pd.DataFrame({'adjectives': words, 'Freq': words_frequency})

print(pd_adjs.head())

# Plot barplots words

sns.set(rc={'figure.figsize':(13,9)})
sns.set(font_scale = 3)
g = sns.barplot(x="adjectives", y="Freq", data=pd_adjs)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
fig = g.get_figure()
fig.savefig(os.path.join(FOLDER_FIGURES, 'total_adjectives.png'))

# plt.bar(y_pos, words_frequency, align='center')
# plt.xticks(y_pos, words, rotation=30)
# plt.show()



# print(definition_text[1:40])


