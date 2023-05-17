"""
Created 17 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different functions to measure the similarity/distance of two words
"""

from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load("en_core_web_sm")


def spacy_similarity(w1, w2):
    # TODO: pos tagging does not work correctly, maybe use a mapping of all tags to specific pos categories (n, v, etc.)
    tokens = nlp(" ".join([w1, w2]))
    print(tokens[0].pos_)
    print(tokens[1].pos_)
    return tokens[0].similarity(tokens[1]) * (tokens[0].pos_ == tokens[1].pos_)


def wordnet_similarity(w1, w2):
    # TODO: WordNet pos tagging works better that spacy, could be used as the general pos tagger
    w1, w2 = wn.synsets(w1)[0], wn.synsets(w2)[0]
    print(w1.pos())
    print(w2.pos())
    return w1.path_similarity(w2) * (w1.pos() == w2.pos())


def mixed_similarity(w1, w2):
    return (spacy_similarity(w1, w2) + wordnet_similarity(w1, w2)) / 2


def main():
    word1 = "clever"
    word2 = "charismatic"

    print("Spacy Similarity:", spacy_similarity(word1, word2))
    print("WordNet Similarity:", wordnet_similarity(word1, word2))
    print("Mixed Similarity:", mixed_similarity(word1, word2))


if __name__ == "__main__":
    main()


