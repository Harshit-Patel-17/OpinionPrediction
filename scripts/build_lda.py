from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

def process_docs(docs):
	texts = []
	print len(docs)
	for doc in docs:

		#Cleaning and tokenizing documents
		raw = doc.lower()
		tokens = tokenizer.tokenize(raw)

		#Remove stop-words from tokens
		stopped_tokens = [token for token in tokens if not token in en_stop]

		#Stem tokens
		stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]

		#Add tokens to list
		texts.append(stemmed_tokens)

	return texts


tokenizer = RegexpTokenizer(r'\w+')

#Create English stop-words list
en_stop = get_stop_words('en')

#Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

#Create documents
politician_docs = []
topic_docs = []

with open("../corpus/PoliticiansCorpora.txt", "r") as infile:
	for line in infile:
		politician_docs.append(line)

with open("../corpus/TopicsCorpora.txt", "r") as infile:
	for line in infile:
		topic_docs.append(line)

#List of tokenized documents
politician_texts = process_docs(politician_docs)
topic_texts = process_docs(topic_docs)

#Create dictionaries
politician_dictionary = corpora.Dictionary(politician_texts)
topic_dictionary = corpora.Dictionary(topic_texts)

#Convert to document-term matrix
politician_corpus = [politician_dictionary.doc2bow(text) for text in politician_texts]
topics_corpus = [topic_dictionary.doc2bow(text) for text in topic_texts]

#Generate LDA model
politician_ldamodel = gensim.models.ldamodel.LdaModel(politician_corpus, num_topics=100, id2word=politician_dictionary, passes=20)
topic_ldamodel = gensim.models.ldamodel.LdaModel(topics_corpus, num_topics=100, id2word=topic_dictionary, passes=20)

politician_dictionary.save("../saved_models/PoliticianDictionary")
topic_dictionary.save("../saved_models/TopicDictionary")
politician_ldamodel.save("../saved_models/PoliticianLdaModel")
topic_ldamodel.save("../saved_models/TopicLdaModel")

