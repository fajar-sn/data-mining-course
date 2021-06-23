import re
import string
import pandas

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def preprocess_data(this_data):
    this_data = this_data.lower()
    this_data = re.sub(r"\d+", "", this_data)
    this_data = this_data.translate(str.maketrans("", "", string.punctuation))
    this_data = this_data.strip()
    return this_data


def filter_data(this_keywords):
    stop_word_list = set(stopwords.words('indonesian'))
    temporary_string = []

    for this_keyword in this_keywords:
        if this_keyword not in stop_word_list:
            temporary_string.append(this_keyword)

    this_keywords = temporary_string
    return this_keywords


def stem_data(this_keywords):
    temporary_string = []
    porter_stemmer = PorterStemmer()
    for this_keyword in this_keywords:
        temporary_string.append(porter_stemmer.stem(this_keyword))
    this_keywords = temporary_string
    return this_keywords


def get_score(this_index):
    filename = f"assets/textmining/news_dataset/data{this_index}.txt"
    file = open(filename, encoding="utf8")
    data = file.read()
    file.close()
    data = preprocess_data(data)
    keywords = word_tokenize(data)
    keywords = filter_data(keywords)
    keywords = stem_data(keywords)
    term_frequency = FreqDist(keywords)
    most_common_keywords = term_frequency.most_common()
    most_frequent_keyword, most_frequent_keyword_frequency = most_common_keywords[0]
    threshold = most_frequent_keyword_frequency / 2
    this_score = []

    for j in range(len(most_common_keywords)):
        if most_common_keywords[j][1] >= threshold:
            this_score.append(most_common_keywords[j])

    return this_score


scores = []

for index in range(50):
    scores.append(get_score(index + 1))

queryList = "pertumbuhan ekonomi, perkembangan pasar dan pergerakan harga saham"
queryList = preprocess_data(queryList)
queryList = word_tokenize(queryList)
queryList = filter_data(queryList)
queryList = stem_data(queryList)

indexedDocs = {}

for i in range(len(queryList)):
    for j in range(len(scores)):
        temp_score = []

        for keyword in scores[j]:
            if keyword[0] == queryList[i]:
                temp_score.append(keyword)
                break

        if len(temp_score) != 0:
            if j not in indexedDocs.keys():
                indexedDocs[j] = temp_score
            else:
                for item in temp_score:
                    indexedDocs[j].append(item)

score_docs = {}

for key, value in indexedDocs.items():
    max_value = 0

    for item in value:
        max_value += item[1]

    score_docs[key] = max_value

ranked_docs = {k: v for k, v in sorted(score_docs.items(), key=lambda x: x[1], reverse=True)}

label = pandas.read_csv("assets/textmining/label.csv", header=None)

temp_data = {}

for key in ranked_docs.keys():
    temp_data[f"data{key}"] = indexedDocs[key]

ranked_docs = temp_data


def count_precision_recall():
    this_recall = []
    this_precision = []
    relevant_retrieved = 0

    for this_key, this_value in ranked_docs.items():
        temp_label = label[label[1] == "ekonomi"]

        for _, df_value in temp_label[0].items():
            if this_key == df_value:
                relevant_retrieved += 1

        this_precision.append(relevant_retrieved / len(ranked_docs))
        this_recall.append(relevant_retrieved / len(label))

    return this_recall, this_precision


recall, precision = count_precision_recall()

print("\nRecall:", recall)
print("\nPrecision:", precision)

import matplotlib.pyplot as plt
import numpy

x = numpy.array(recall)
y = numpy.array(precision)

plt.plot(x, y)
plt.show()
