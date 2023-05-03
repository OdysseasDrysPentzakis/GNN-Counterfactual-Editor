"""
Created 3 May 2023
@author: Dimitris Lymperopoulos
Description: A script to test the functionality of the ConceptNetConnector class

Usage:
1)  Obtain information about what each method of the connector returns, and how the data are
    being represented by giving a sample term to search.
    python3 test_CNConnector.py
        --search <a string to be searched in the ConceptNet>

2)  Obtain information about the class and its methods using the default search_term, which is
    the word <dog>.
    python3 test_CNConnector.py

Example:
    python3 test_CNConnector.py
        --search table

"""

import argparse
from datetime import datetime
from Connectors.ConceptNetConnector import ConceptNetConnector


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

    replacements = connector.find_replacements(args.search, quantity=5, synonym=False)
    print("\nThe method find_replacements() returns a list of possible replacements for the given term")
    print("In this case, for the term {} the first 5 replacements are:".format(args.search))
    print("\n".join(replacements))

    print("\n\nScript execution time: " + str(datetime.now()-start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)