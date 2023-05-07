"""
Created 3 May 2023
@author: Dimitris Lymperopoulos
Description: A script containing a custom connector class for interaction with WordNet
"""

from nltk.corpus import wordnet as wn


class WordNetConnector:
    def __init__(self):
        pass

    def find_replacements(self, term, quantity=1, synonym=False, pos=None):
        synsets = wn.synsets(term)
        replacements = []

        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.antonyms() and not synonym and (lemma.synset().pos() == pos or pos is None):
                    replacements.append(lemma.antonyms()[0].name())

                if synonym:
                    for syn in wn.synsets(lemma.name(), pos=pos):
                        replacements.extend(lem.name() for lem in syn.lemmas() if lem.name() != term)

        return replacements if quantity >= len(replacements) else replacements[:quantity]
