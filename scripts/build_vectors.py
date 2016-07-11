#Log various events
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Build vectors
from gensim import corpora, models, similarities
documents = models.doc2vec.TaggedLineDocument("../corpus/PoliticiansCorpora.txt")
model = models.doc2vec.Doc2Vec(documents, size=300)
model.save("../saved_models/PoliticianCorporaVectors.txt")

documents = models.doc2vec.TaggedLineDocument("../corpus/TopicsCorpora.txt")
model = models.doc2vec.Doc2Vec(documents, size=300)
model.save("../saved_models/TopicsCorporaVectors.txt")