import re
import string
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plot
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

f = open("assets/news.txt", "r")
text = f.read()
f.close()

print("\nText:\n-------------------------------------------------------------------------------------------\n", text)

text = text.lower()
# print("\nHuruf kecil semua:\n------------------------------------------------------------------------------\n", text)

text = re.sub(r"\d+", "", text)
# print("\nAngka hilang:\n-----------------------------------------------------------------------------------\n", text)

text = text.translate(str.maketrans("", "", string.punctuation))
# print("\nTanda baca hilang:\n------------------------------------------------------------------------------\n", text)

text = text.strip()
# print("\nKarakter kosong hilang:\n-------------------------------------------------------------------------\n", text)

tokens = word_tokenize(text)
# print("\nTokenizing:\n-----------------------------------------------------------------------------------\n", tokens)

# # Filtering dengan Sastrawi
# factory = StopWordRemoverFactory()
# stopword = factory.create_stop_word_remover()
# text = stopword.remove(text)
# # print("\nSetelah filtering:\n----------------------------------------------------------------------------\n", text)
#
# # Stemming dengan Sastrawi
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# text = stemmer.stem(text)
# print("\nOutput stemming:\n--------------------------------------------------------------------------------\n", text)

# Filtering dengan Porter
listStopword = set(stopwords.words('indonesian'))
tmpstr = []
for t in tokens:
    if t not in listStopword:
        tmpstr.append(t)

tokens = tmpstr
print("\nSetelah filtering:\n----------------------------------------------------------------------------\n", tokens)

# Stemming dengan Porter
tmpstr = []
ps = PorterStemmer()
for k in tokens:
    tmpstr.append(ps.stem(k))
tokens = tmpstr
print("\nOutput stemming:\n------------------------------------------------------------------------------\n", tokens)

tf = FreqDist(tokens)
print("\nTerm Frequency:\n---------------------------------------------------------------------\n", tf.most_common())

word, frequency = tf.most_common()[0]
print("\nKeyword yang paling banyak muncul:\n----------------------------------------\n", word, "=", frequency, "\n")

print("\nKeseluruhan keywords:\n---------------------------------------------------------------------------------\n")
for word, frequency in tf.most_common():
    print(word, ":", frequency)

# tf.plot(cumulative=False)
# plot.show()
