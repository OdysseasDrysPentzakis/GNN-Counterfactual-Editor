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
        """
        A method that takes as input a string and an integer and returns a list of possible replacements
        for that string, based on the WordNet graph

        :param term: the string that needs to be replaced
        :param quantity: number of candidate replacements to be returned
        :param synonym: a boolean value indicating if the given term should be replaced with synonyms or antonyms
        :param pos: a string representing the part-of-speach of the replacements (and the given term)
        :return: a list with length=quantity, with candidate replacements for the given term
        """
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
