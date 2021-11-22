import nltk
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pandas as pd
import openpyxl
import numpy as np
# from nltk.tag import pos_tag
# import nltk
import re
from nltk.corpus import wordnet
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from wordfreq import word_frequency, zipf_frequency
import dale_chall

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('wordnet')

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

dtypes = {"corpus": "string", "sentence": "string", "token": "string", "complex": "float64"}
train = pd.read_excel('train.xlsx', dtype=dtypes, keep_default_na=False)
test = pd.read_excel('test.xlsx', dtype=dtypes, keep_default_na=False)
print("DATA READ"),

train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(train, train['complex'].values, test_size=0.3)
print("DATA SPLITTED")


def corpus_to_int(corpus):
    if corpus == 'bible':
        return np.array(0)
    if corpus == 'biomed':
        return np.array(1)
    return np.array(2)


def word_def_len(word):
    nr = 0
    defs = wordnet.synsets(word.lower())

    return len(defs)
    # if len(defs) == 0:
    #     return np.array(-1)
    #
    # for definition in defs:
    #     nr += len(definition.definition().split(' '))
    #
    # return np.array(nr / len(defs))


#         for nlen in range(3, min(4, len(word))):
#             for ii in range(len(word) - nlen + 1):


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


def xyzh(word):
    nr = 0
    nr += word.lower().count('x')
    nr += word.lower().count('y')
    nr += word.lower().count('z')
    nr += word.lower().count('h')

    return np.array(nr)


def consoane(word):
    nr = 0
    nr += word.lower().count('a')
    nr += word.lower().count('e')
    nr += word.lower().count('i')
    nr += word.lower().count('o')
    nr += word.lower().count('u')

    return np.array(len(word) - nr)


def preload(rows, vocabular, ngram_list):
    features = []
    # features.append(np.array(['corpus', 'def', 'consecutive', 'CAPS', 'Title', 'Length', 'xyzh', 'consoane', 'vocale', 'frecventa']))
    for index, row in rows.iterrows():
        features.append(featureise(row, vocabular, ngram_list))

    return np.array(features)


def process_tokens(lista):
    list_grams = []

    for word in lista:
        if len(word) > 2:
            for nlen in range(3, min(4, len(word))):
                for ii in range(len(word) - nlen + 1):
                    list_grams.append(word[ii:(ii + nlen)])

    # trebuie sa o fac de fapt map, sa aibe aparitii unice
    return list_grams


def best_ngram_letters(word, lista):
    max = 0
    ngram = ''
    low_word = word.lower()

    if len(word) > 2:
        for nlen in range(3, min(4, len(word))):
            for ii in range(len(word) - nlen + 1):
                aux = lista.count(low_word[ii:(ii + nlen)])
                # print(aux, low_word[ii:(ii + nlen)])
                if aux > max:
                    max = aux
                    ngram = low_word[ii:(ii + nlen)]

    return np.array(abs(hash(ngram)) % 1000)
    # abs(hash(


def map_words(list):
    list_aux = []
    for elem in list:
        sens = remove_punct(elem)
        list_aux.extend(sens.split())

    return list_aux


# remove punctuation from sentence
def remove_punct(sens):
    sens = re.sub(r'[^\w\s]', ' ', sens)
    return ' '.join(sens.split())


def generateFile(train, test, preds, file_name):
    df = pd.DataFrame()
    # df['id'] = test.index + len(train) + 1
    df['id'] = test.index + len(train) + 1
    df['complex'] = preds
    df.to_csv(file_name, index=False)


def MLPpred(train_features, train_classes, test_features):
    # identity, logistic, tanh, relu  # adam, sgd, lbfgs                                                            # constant, invscaling
    clf = MLPClassifier(hidden_layer_sizes=(200,), activation='logistic', solver='lbfgs', alpha=0.00001,
                        max_iter=100000, n_iter_no_change=100)
    # relu, lbfgs, 0.01
    # 0.72: hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', alpha=0.0769696969, max_iter=10000, n_iter_no_change=100, learning_rate_init=0.69, learning_rate='invscaling'
    # 0.72: hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', alpha=0.01, max_iter=10000, n_iter_no_change=100
    # 0.71: (hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', alpha=0.03, max_iter=10000, n_iter_no_change=100, learning_rate_init=0.02, learning_rate='invscaling')
    # 0.7: hidden_layer_sizes=(300,), activation='logistic', solver='lbfgs', alpha=0.03, max_iter=10000, n_iter_no_change=100, learning_rate_init=0.05, learning_rate='invscaling'
    return getPrediction(clf, train_features, train_classes, test_features)


def SVMpred(c, g, train_features, train_classes, test_features):
    clf = svm.SVC(C=c, gamma=g)

    return getPrediction(clf, train_features, train_classes, test_features)


def NaiveBaisePred(train_features, train_classes, test_features):
    clf = GaussianNB()

    return getPrediction(clf, train_features, train_classes, test_features)


def KnnPred(k, train_features, train_classes, test_features):
    clf = KNeighborsClassifier(n_neighbors=k)

    return getPrediction(clf, train_features, train_classes, test_features)


def getPrediction(clf, train_features, train_classes, test_features):
    clf.fit(train_features, train_classes)
    predictions = clf.predict(test_features)

    return predictions


def getAccuracy(preds, test_y):
    cfm_validate = confusion_matrix(test_y, preds)
    print(cfm_validate)
    print(balanced_accuracy_score(test_y, preds))


def normalizare(wordlist_features):
    normalized_f = list()
    for f_set in wordlist_features:
        aux_list = list()
        for f_val in f_set:
            aux_list.append((f_val - min(f_set)) / (max(f_set) - min(f_set)))
        normalized_f.append(aux_list)

    return normalized_f


def count_syllables(word):
    vowel_runs = len(VOWEL_RUNS.findall(word))
    exceptions = len(EXCEPTIONS.findall(word))
    additional = len(ADDITIONAL.findall(word))
    return max(1, vowel_runs - exceptions + additional)


VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)
EXCEPTIONS = re.compile(
    # fixes trailing e issues:
    # smite, scared
    "[^aeiou]e[sd]?$|"
    # fixes adverbs:
    # nicely
    + "[^e]ely$",
    flags=re.I
)
ADDITIONAL = re.compile(
    # fixes incorrect subtractions from exceptions:
    # smile, scarred, raises, fated
    "[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|"
    # fixes miscellaneous issues:
    # flying, piano, video, prism, fire, evaluate
    + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",
    flags=re.I
)


def holonime(word):
    aux = wordnet.synsets(word.lower())
    if len(aux) > 0:
        print(aux)
        print(aux[0], len(aux[0].substance_holonyms()))
        return np.array(len(aux[0].substance_holonymssubstance_holonyms()))


def featureise(row, vocabular, ngram_list):
    features = []

    # features.append(ngram_letters(row['token']))
    features.append(corpus_to_int(row['corpus']))
    # features.append(word_def_len(row['token']))
    features.append(
        np.array(int(row['token'] in dale_chall.DALE_CHALL or row['token'].lower() in dale_chall.DALE_CHALL)))
    features.append(np.array(count_syllables(row['token'])))
    features.append(np.array(len(row['token']) - consoane(row['token'])))
    # features.append(np.array(consoane(row['token'])))
    features.append(consecutive_consonants(row['token']))
    # features.append(np.array(int(row['token'].isupper())))  # check if it's only CAPS
    features.append(np.array(int(row['token'].istitle())))  # check if it's a title
    features.append(np.array(len(row['token'])))  # word len

    # features.append(xyzh(row['token']))
    # features.append(np.array(row['token'].count('x')))  # how many x
    # features.append(np.array(row['token'].count('y')))  # how many y
    # features.append(np.array(row['token'].count('z')))  # how many z
    # features.append(np.array(int(row['token'].count('h') > 0)))  # how many h
    # features.append(np.array(vocabular.count(row['token'])))  # de cate ori apare in toate propozitiile
    features.append(np.array(zipf_frequency(row['token'].lower(), 'en')))
    features.append(np.array(word_frequency(row['token'].lower(), 'en')))
    # features.append(best_ngram_letters(row['token'], ngram_list))  # scoate cea mai comuna secventa de litere
    # features.append(np.array(sum(len(sens.hypernyms()) for sens in wordnet.synsets(row['token'].lower()))))  # cate sensuri au min 1 hipernim
    # features.append(np.array(sum(len(sens.hyponyms()) for sens in wordnet.synsets(row['token'].lower())) / (len(wordnet.synsets(row['token'].lower())) + 1)))  # suma hiponimelor tuturor sensurilor

    # features.append(holonime(row['token']))

    return np.array(features)


# ngram_list = process_tokens(train['token'].values)
ngram_list = []

# vocabular = map_words(train['sentence'].values)
vocabular = []

# features = preload(train_x, vocabular, ngram_list)
# test_x_features = preload(test_x, vocabular, ngram_list)
#
# # preds = KnnPred(7, features, train_y, test_x_features)
# # getAccuracy(preds, test_y)
# # def_len, consec, consoane, vocabular.count
# preds = NaiveBaisePred(features, train_y, test_x_features)
# getAccuracy(preds, test_y)
# # 12, 1.1
# #
# # for i in range(1, 1000, 11):
# #     print(i)
# #     preds = SVMpred(i, 2, features, train_y, test_x_features)
# #     getAccuracy(preds, test_y)
# # #
# # preds = MLPpred(features, train_y, test_x_features)
# # getAccuracy(preds, test_y)


#

features = preload(train, vocabular, ngram_list)
test_x_features = preload(test, vocabular, ngram_list)
#
# # preds = KnnPred(7, features, train['complex'].values, test_x_features)
# #
# preds = NaiveBaisePred(features, train['complex'].values, test_x_features)
# #
# preds = SVMpred(1105, 0.1, features, train['complex'].values, test_x_features)
#

# # preds = MLPpred(features, train['complex'].values, test_x_features)
#
# generateFile(train, test, preds, 'submission-naive-bayes-101-selaru.csv')
