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
        """

        :param pos: a string representing the part-of-speach that the replacements will occur
        :param synonyms: a boolean value indicating if replacements shoud be synonyms or antonyms of the original words
        """
        # create initial connector objects for interaction with WordNet and ConceptNet
        self.cnc = ConceptNetConnector(conceptnet_api=False,
                                       conceptnet_db_path=os.path.join(os.getcwd(), os.pardir, "Connectors")
                                       )
        self.wnc = WordNetConnector()
        # pos - Part of Speech (verb, noun, etc.)
        self.pos = pos
        # synonyms - if replacements should be synonyms or antonyms
        self.synonyms = synonyms

    def get_replacements(self, word, word_similarity, pos=None):
        """
        A function that searches the ConceptNet in order to find suitable replacements for the given word. If such
        replacements do not exist, then it searches the WordNet for them.

        :param word: a string representing the original word that need to be replaced
        :param word_similarity: a function that returns a similarity score between two words
        :param pos: a string representing part-of-speach of the original word and the replacements
        :return: a list of possible replacements for the given word
        """
        # get possible replacements using ConceptNet, and if none were found, use WordNet instead
        replacements = self.cnc.find_replacements(word, quantity=5, synonym=self.synonyms)
        if len(replacements) == 0:
            replacements = self.wnc.find_replacements(word, quantity=5, synonym=self.synonyms, pos=pos)

        # return the candidate with the maximum similarity with the original word
        return max(replacements, key=lambda x: word_similarity(word, x))

    def generate_counterfactual(self, original_sentence, indicative_sentence, word_similarity):
        """
        A function that takes as input an original_sentence, and based on a given indicative_sentence, uses knowledge
        extracted from ConceptNet and WordNet in order to generate a counterfactual sentence.

        :param original_sentence: a string representing the sentence whose words will be changed
        :param indicative_sentence: a string that indicated which words of the original sentence shall be changed
        :param word_similarity: a function that returns a similarity score between two words
        :return: the original sentence where the specified words have been replaced with those extracted from ConceptNet
        """
        original_sentence_list = original_sentence.split()
        indicative_sentence_list = indicative_sentence.split()

        # change the desired words with replacements based on ConceptNet and WordNet connections
        for i in range(len(original_sentence_list)):
            if original_sentence_list[i] == indicative_sentence_list[i]:
                continue
            else:
                indicative_sentence_list[i] = self.get_replacements(original_sentence_list[i], self.pos,
                                                                    word_similarity)

        return " ".join(indicative_sentence_list)
