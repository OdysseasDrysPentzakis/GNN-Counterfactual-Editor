"""
Created 2 May 2023
@author: Dimitris Lymperopoulos
Description: A script containing a custom connector class with the ConceptNet api

Usage:
1)  Obtain information about what each method of the connector returns, and how the data are
    being represented by giving a sample term to search.
    python3 ConceptNetConnector.py
        --search <a string to be searched in the ConceptNet>

2)  Obtain information about the class and its methods using the default search_term, which is
    the word <dog>.
    python3 ConceptNetConnector.py

Example:
    python3 ConceptNetConnector.py
        --search table
"""

import requests
import argparse
from datetime import datetime


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

    # TODO: implement the decision process for the optimal replacements
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

        positive_edges = {'Synonym', 'RelatedTo'}
        negative_edges = {'RelatedTo'}

        # we also filter the edges, to keep only those which connect two english terms
        for e in filter(lambda x: '/en/' in x[1] and '/en/' in x[2], self.find_connected_edges(term)):
            if e[0][3:-1] in positive_edges if synonym else negative_edges:
                replacements.append(e[1][4:-1] if e[1][4:-1] != term else e[2][4:-1])

        return replacements[:quantity]


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
        """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--search", type=str, action="store", metavar="search",
                        required=False, default="dog", help="The term to be searched in the ConceptNet")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    connector = ConceptNetConnector()
    reply = connector.search_term(args.search)

    print("The returning object of the search_term() method has the following keys:")
    print(", ".join(k for k in reply.keys()))
    print("\nThe most important field is the 'edges', which contains the different labeled edges that connect the "
          "current word with other terms")

    edges = [e['@id'] for e in reply['edges']]
    print("The edges are represented as below:")
    print("\n".join(edges))

    print("\nThe type of each edge is:")
    print(type(edges[0]))

    connected_edges = connector.find_connected_edges(args.search)
    print("\nThe returned object from the find_connected_edges() method is the list of edges below:")
    print("\n".join(" ".join(ce) for ce in connected_edges))

    print('\nThe type of the each edge returned from the last method is:')
    print(type(connected_edges[0]))

    replacements = connector.find_replacements(args.search, quantity=5, synonym=True)
    print("\nThe method find_replacements() returns a list of possible replacements for the given term")
    print("In this case, for the term {} the first 5 replacements are:".format(args.search))
    print("\n".join(replacements))


    print("\n\nScript execution time: " + str(datetime.now()-start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
