"""
Created 2 May 2023
@author: Dimitris Lymperopoulos
Description: A script containing a custom connector class with the ConceptNet api
"""

import requests


class ConceptNetConnector:
    def __init__(self):
        self.node_uri = "/c/en/"
        self.edge_uri = "/r/"
        self.api_url = "http://api.conceptnet.io"

    def search_term(self, term):
        """
        :param term: string indicating the term to be searched in the ConceptNet
        :return: a json object containing the response from the ConceptNet api
        """
        return requests.get(self.api_url + self.node_uri + term).json()

    def find_connected_edges(self, term):
        """
        A method that takes as input a term, and uses the response from the ConceptNet api to return the connected
        edges to that term

        :param term: a string indicating the term whose edges will be searched
        :return: a list of edges connected to the search term
        """
        connected_edges = [e['@id'][4:-1].split(',') for e in self.search_term(term)['edges']]

        return connected_edges

    def find_replacements(self, term, quantity=1, synonym=False):
        """
        A method that takes as input a string and an integer and returns a list of possible replacements
        for that string, based on labels of connected to that string edges in the ConceptNet graph.

        :param term: the string that needs to be replaced
        :param quantity: number of candidate replacements to be returned
        :param synonym: a boolean value indicating if the given term should be replaced with synonyms or antonyms
        :return: a list with length=quantity, with candidate replacements for the given term
        """

        replacements = []
        positive_edges = {'Synonym', 'SimilarTo', 'IsA'}
        negative_edges = {'Antonym'}

        # we also filter the edges, to keep only those which connect two english terms
        for e in filter(lambda x: '/en/' in x[1] and '/en/' in x[2], self.find_connected_edges(term)):
            if e[0][3:-1] in (positive_edges if synonym else negative_edges):
                replacements.append(e[1].split('/')[3] if e[1].split('/')[3] != term else e[2].split('/')[3])

        return replacements if quantity >= len(replacements) else replacements[:quantity]
