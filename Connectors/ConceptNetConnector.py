"""
Created 2 May 2023
@author: Dimitris Lymperopoulos
Description: A script containing a custom connector class with the ConceptNet api

Usage:
    Obtain information about what each method of the connector returns, and how the data are
    being represented.
    python3 ConceptNetConnector.py
        --search <a string to be searched in the ConceptNet>

Example:
    python3 ConceptNetConnector.py
        --search dog
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

    # TODO: determine how to replace the given term from the returned edges
    def replace_term(self, term, synonym=False):
        """
        A function that takes as input a term, and uses the response from the ConceptNet api to determine a proper
        replacement for that term.

        :param term: a string indicating the term that needs to be replaced
        :param synonym: a boolean value determining if the replacement will be synonym or antonym of the term
        :return: a string representing the replacement for the original term
        """
        connected_edges = [e['@id'] for e in self.search_term(term)['edges']]
        return connected_edges


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
        """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--search", type=str, action="store", metavar="search",
                        required=True, help="The term to be searched in the ConceptNet")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    connector = ConceptNetConnector()
    reply = connector.search_term(args.search)

    print("The returning object has the following keys:")
    print(", ".join(k for k in reply.keys()))
    print("\nThe most important field is the 'edges', which contains the different labeled edges that connect the "
          "current word with other terms")

    print("The edges are represented as below:")
    edges = [e['@id'] for e in reply['edges']]
    print("\n".join(edges))

    print("\nThe type of each edge is:")
    print(type(edges[0]))

    print("\nScript execution time: " + str(datetime.now()-start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)




