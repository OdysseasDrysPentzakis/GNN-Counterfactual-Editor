"""
Created 3 May 2023
@author: Dimitris Lymperopoulos
Description: A script to test the functionality of the WordNetConnector class

Usage:
1)  Obtain information about what each method of the connector returns, and how the data are
    being represented by giving a sample term to search and a part-of-speech that filters the
    replacements that are returned
    python3 test_WNConnector.py
        --search <a string to be searched in the WordNet>
        --pos <'n' (noun) or 'v' (verb) or 'a' (adjective) or 'r' (adverb) or 's' (satellite adjective)>

2)  Obtain information about the class and its methods using the default search_term, which is
    the word <clever> and the default part-of-speech, which is 'a' (adjective)
    python3 test_WNConnector.py

Example:
    python3 test_WNConnector.py
        --search pretty
        --pos a

"""

import argparse
from datetime import datetime
from Connectors.WordNetConnector import WordNetConnector


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
        """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--search", type=str, action="store", metavar="search",
                        required=False, default="clever", help="The term to be searched in the ConceptNet")
    parser.add_argument("-p", "--pos", choices=['n', 'v', 'a', 'r', 's'], action='store', metavar="pos",
                        required=False, default='s', help="The part-of-speech that the replacements should be")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    connector = WordNetConnector()

    replacements = connector.find_replacements(args.search, quantity=5, synonym=True, pos=args.pos)
    print("\nThe method find_replacements() returns a list of possible replacements for the given term")
    print("In this case, for the term {} the first 5 replacements are:".format(args.search))
    print("\n".join(replacements))

    print("\n\nScript execution time: " + str(datetime.now()-start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
