import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pandas as pd
# import openpyxl
import numpy as np
from PyDictionary import PyDictionary
# from nltk.tag import pos_tag
# import nltk
import re
from nltk.corpus import wordnet

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

dictionary = PyDictionary()

dtypes = {"corpus": "string", "sentence": "string", "token": "string", "complex": "float64"}
train = pd.read_excel('train.xlsx', dtype=dtypes, keep_default_na=False)
test = pd.read_excel('test.xlsx', dtype=dtypes, keep_default_na=False)
print("DATA READ"),

train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(train, train['complex'].values, test_size=0.3)
print("DATA SPLITTED")


def corpus_to_int(corpus):
    if corpus == 'bible':
        return np.array(1)
    if corpus == 'biomed':
        return np.array(2)
    return np.array(3)


# remove punctuation from sentence
def remove_punct(sens):
    sens = re.sub(r'[^\w\s]', ' ', sens)
    return ' '.join(sens.split())


# the avg number of words in the definitions of the word
# def word_def_len(sens, word):
#     # dictionary.meaning("indentation")['Noun'][0]
#     # pos_tag(word_tokenize("John's big idea isn't all that bad."), tagset='universal')
#     # .lower().capitalize()
#
#     # getting the type of the word: verb, noun, adjective
#     # sens = remove_punct(sens)
#     #
#     # word_type = pos_tag(word_tokenize(sens), tagset='universal')[sens.split(' ').index(word)][1].lower().capitalize()
#     # if word_type == 'Adj':
#     #     word_type = 'Adjective'
#     # elif word_type == 'Adv':
#     #     word_type = 'Adverb'
#     # elif word_type == 'Adp':
#     #     word_type = 'Preposition'
#     # elif word_type == 'Conj':
#     #     word_type = 'Conjunction'
#
#     try:
#         meanings = dictionary.meaning(word)
#         nr = 0
#         for type in meanings:
#             for mean in meanings[type]:
#                 nr += len(mean.split(' '))
#         return np.array(nr / len(meanings))
#     except (TypeError, KeyError):
#         return np.array(-1)

def word_def_len(sens, word):
    nr = 0
    defs = wordnet.synsets(word)

    if len(defs) == 0:
        return np.array(-1)

    for definition in defs:
        nr += len(definition.definition().split(' '))

    return np.array(nr / len(defs))


def ngram_letters(word):
    list_grams = []
    # print(word, type(word))
    # generate of the n-grams but for the letters of the words
    for nlen in range(2, min(4, len(word))):
        for ii in range(len(word) - nlen + 1):
            list_grams.append(word[ii:(ii + nlen)])
    return np.array(list_grams)


# return the maximum number of consecutive consonants in the word
def consecutive_consonants(word):
    word = word.lower()
    consec_conso = 0
    max = 0
    # Start traversing the string
    for letter in word:

        # Check if current character is
        # vowel or consonant
        if letter in 'aeiou':
            if consec_conso > max:
                max = consec_conso
            consec_conso = 0

        # Increment counter for
        # consecutive consonants
        else:
            consec_conso += 1

    if consec_conso > max:
        max = consec_conso

    return np.array(max)


def featureise(row):
    features = []

    features.append(corpus_to_int(row['corpus']))
    features.append(word_def_len(row['sentence'], row['token']))
    # features.append(ngram_letters(row['token']))
    features.append(consecutive_consonants(row['token']))
    features.append(int(row['token'].isupper()))  # check if it's only CAPS

    return np.array(features)


def preload(rows):
    features = []
    for index, row in rows.iterrows():
        features.append(featureise(row))

    return np.array(features)


features = preload(train_x)
test_x_features = preload(test_x)
#region KNN
# 1
# clf = KNeighborsClassifier(n_neighbors=1)
#endregion

#region Naive
clf = GaussianNB()
#endregion

clf.fit(features, train_y)
# y_pred = clf.predict(test_images)

preds = clf.predict(test_x_features)

# features = []
# return np.array(features) chiar daca e un singur lucru

cfm_validate = confusion_matrix(test_y, preds)
print(cfm_validate)

print(((cfm_validate[0][0] / (cfm_validate[0][0] + cfm_validate[0][1])) + (
        cfm_validate[1][1] / (cfm_validate[1][1] + cfm_validate[1][0]))) / 2)


#region SVM
# for c in range(1, 1000, 50):
#     # c = 201, gamma=1
#     clf = svm.SVC(C=c, gamma=1)
#endregion



#region Submit SVM
# features = preload(train)
# test_x_features = preload(test)
#
# # for c in range(1, 1000, 50):
#     # c = 201, gamma=1
# clf = svm.SVC(C=201, gamma=1)
# clf.fit(features, train['complex'].values)
#
# # print(c)
# # y_pred = clf.predict(test_images)
#
# preds = clf.predict(test_x_features)
#
# df = pd.DataFrame()
# # df['id'] = test.index + len(train) + 1
# df['id'] = test.index + len(train) + 1
# df['complex'] = preds
# df.to_csv('submission-svm.csv', index=False)
#endregion