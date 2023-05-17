"""
Created 7 May 2023
@author: Dimitris Lymperopoulos
Description: A script containing a dummy counterfactual editor class that uses ConceptNet and WordNet
"""
import os

from Connectors.ConceptNetConnector import ConceptNetConnector
from Connectors.WordNetConnector import WordNetConnector


class DummyEditor:
    def __init__(self, pos=None, synonyms=False):
        # create initial connector objects for interaction with WordNet and ConceptNet
        self.cnc = ConceptNetConnector(conceptnet_api=False,
                                       conceptnet_db_path=os.path.join(os.getcwd(), os.pardir, "Connectors")
                                       )
        self.wnc = WordNetConnector()
        # pos - Part of Speech (verb, noun, etc.)
        self.pos = pos
        # synonyms - if replacements should be synonyms or antonyms
        self.synonyms = synonyms

    def get_replacements(self, word, pos=None):
        # get possible replacements using ConceptNet, and if none were found, use WordNet instead
        replacements = self.cnc.find_replacements(word, quantity=5, synonym=self.synonyms)
        if len(replacements) == 0:
            replacements = self.wnc.find_replacements(word, quantity=5, synonym=self.synonyms, pos=pos)

        # return the candidate with the maximum similarity with the original word - TODO
        return replacements[0]

    def generate_counterfactual(self, original_sentence, indicative_sentence):
        original_sentence_list = original_sentence.split()
        indicative_sentence_list = indicative_sentence.split()

        # change the desired words with replacements based on ConceptNet and WordNet connections
        for i in range(len(original_sentence_list)):
            if original_sentence_list[i] == indicative_sentence_list[i]:
                continue
            else:
                indicative_sentence_list[i] = self.get_replacements(original_sentence_list[i], self.pos)

        return " ".join(indicative_sentence_list)
