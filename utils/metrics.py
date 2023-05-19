"""
Created 17 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different functions to measure the similarity/distance of two words
"""

import os
import pickle
import numpy as np
import spacy
from scipy import spatial
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
    """
    A function that computes the similarity of two words based on their embedding vectors cosine similarity
    from SpaCy.

    :param w1: a string representing the first word
    :param w2: a string representing the second word
    :return: a float representing the similarity of the two words
    """

    tokens = nlp(" ".join([w1, w2]))
    # TODO: handle pos-tags that are not present in the pos_map.keys()
    # TODO: perhaps do not use pos-tagging in the computation of the word similarity for spacy
    return tokens[0].similarity(tokens[1]) * (
            pos_map.get(tokens[0].pos_, tokens[0].pos_) == pos_map.get(tokens[1].pos_, tokens[1].pos_))


def wordnet_similarity(w1, w2):
    """
        A function that computes the similarity of two words based on knowledge from WordNet.

        :param w1: a string representing the first word
        :param w2: a string representing the second word
        :return: a float representing the similarity of the two words
        """

    w1, w2 = wn.synsets(w1)[0], wn.synsets(w2)[0]
    return w1.path_similarity(w2) * (w1.pos() == w2.pos())


def mixed_similarity(w1, w2):
    """
        A function that computes the similarity of two words as the average similarity between their SpaCy-based
        similarity and the WordNet-based similarity.

        :param w1: a string representing the first word
        :param w2: a string representing the second word
        :return: a float representing the similarity of the two words
        """

    return (spacy_similarity(w1, w2) + wordnet_similarity(w1, w2)) / 2


def conceptnet_similarity(w1, w2, embeddings):
    """
        A function that computes the similarity of two words based on their embedding vectors cosine similarity
        from ConceptNet Numberbatch.

        :param w1: a string representing the first word
        :param w2: a string representing the second word
        :param embeddings: a dictionary containing the embedding vectors generated from ConceptNet
        :return: a float representing the similarity of the two words
        """

    return 1 - spatial.distance.cosine(np.array(embeddings[w1], dtype=float), np.array(embeddings[w2], dtype=float))


def main():
    word1 = "dog"
    word2 = "cat"
    with open(os.path.join(os.getcwd(), os.pardir, 'Data', 'cn_embeddings.p'), 'rb') as f:
        cn_embeddings = pickle.load(f)

    print("Spacy Similarity:", spacy_similarity(word1, word2))
    print("WordNet Similarity:", wordnet_similarity(word1, word2))
    print("Mixed Similarity:", mixed_similarity(word1, word2))
    print("ConceptNet Similarity:", conceptnet_similarity(word1, word2, cn_embeddings))


if __name__ == "__main__":
    main()
