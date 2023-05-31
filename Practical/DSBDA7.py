import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Sample document
doc = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."

doc = doc.lower()
doc

import string
print(string.punctuation)
doc = "".join([char for char in doc if char not in string.punctuation])
print(doc)

#Tokenization
nltk.download('punkt')

words = word_tokenize(doc)
print(words)

# stopword removal
nltk.download('stopwords')
stop_words = stopwords.words('english')
print(stop_words)

filtered_words = [word for word in words if word not in stop_words]
print(filtered_words)

# stemming
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in filtered_words]
print(stemmed)

# pos tagging
nltk.download('averaged_perceptron_tagger')

pos = pos_tag(filtered_words)
print(pos)

# lemmatization
nltk.download('wordnet')

# lemmatization
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_words]
print(lemmatized_tokens)

# Document representation using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([doc])

print(tfidf_matrix)
