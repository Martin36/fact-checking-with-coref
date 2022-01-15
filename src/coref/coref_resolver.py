import spacy
nlp = spacy.load("en_core_web_sm")

# TODO: This doesn't work with Spacy V3 apparently
import neuralcoref
neuralcoref.add_to_pipe(nlp)

doc = nlp(u'My sister has a dog. She loves him.')

doc._.has_coref
doc._.coref_clusters
