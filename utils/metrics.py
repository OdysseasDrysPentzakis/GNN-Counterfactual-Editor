"""
Created 17 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different functions to measure the similarity/distance of two words
"""

import spacy
from nltk.corpus import wordnet as wn

# TODO: perhaps use a different (larger) model for better and more accurate results
nlp = spacy.load("en_core_web_sm")

# dictionary that generalizes some specific pos-tags from spacy
pos_map = {
    'PROPN': 'NOUN',
    'PRON': 'NOUN',
    'AUX': 'VERB'
}


def spacy_similarity(w1, w2):
    tokens = nlp(" ".join([w1, w2]))
    # print(pos_map.get(tokens[0].pos_, tokens[0].pos_))
    # print(pos_map.get(tokens[1].pos_, tokens[1].pos_))
    # TODO: handle pos-tags not present in the pos_map.keys()
    # TODO: perhaps do not use pos-tagging in the computation of the word similarity
    return tokens[0].similarity(tokens[1]) * (
            pos_map.get(tokens[0].pos_, tokens[0].pos_) == pos_map.get(tokens[1].pos_, tokens[1].pos_))


def wordnet_similarity(w1, w2):
    w1, w2 = wn.synsets(w1)[0], wn.synsets(w2)[0]
    # print(w1.pos())
    # print(w2.pos())
    return w1.path_similarity(w2) * (w1.pos() == w2.pos())


def mixed_similarity(w1, w2):
    return (spacy_similarity(w1, w2) + wordnet_similarity(w1, w2)) / 2


def main():
    word1 = "dog"
    word2 = "cat"

    print("Spacy Similarity:", spacy_similarity(word1, word2))
    print("WordNet Similarity:", wordnet_similarity(word1, word2))
    print("Mixed Similarity:", mixed_similarity(word1, word2))


if __name__ == "__main__":
    main()


