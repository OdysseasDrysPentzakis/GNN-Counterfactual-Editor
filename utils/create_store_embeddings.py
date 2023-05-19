"""
Created 19 May 2023
@author: Dimitris Lymperopoulos
Description: A script that generates a json file with embeddings of each word based on a source (.txt) file
"""

import os
import gzip
import pickle
import argparse
from datetime import datetime


def create_store_embeddings(src_file, dest_file):
    """
    :param src_file: a filepath to a .txt.gz file containing words and their vectors from ConceptNet Numberbatch
    :param dest_file: a .p filepath representing the file where the pairs (word: vector) will be stored
    :return: None
    """

    # check if src_file exists
    if not os.path.exists(src_file):
        print("[ERROR]: File {} does not exist!".format(src_file))
        exit(1)

    # continue to create the embeddings dictionary
    embeddings = {}
    with gzip.open(src_file, 'rt', encoding='utf8') as f:
        for line in f.readlines():
            # skip words that start with a number or a symbol
            if not str.isalpha(str(line)[0]):
                continue

            word = line.split()[0]
            embeddings[word] = line.split()[1:]

    # store the dictionary in a pickle file
    with open(dest_file, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Word-Vector pairs successfully stored in {}".format(dest_file))


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-file", type=str, action="store", metavar="src_file",
                        required=True, help="A .txt.gz file containing from ConceptNet Numberbatch")
    parser.add_argument("-d", "--dest-file", type=str, action="store", metavar="dest_file",
                        required=True, help="A .p (pickle) file where the word-vector pairs will be stored")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    create_store_embeddings(src_file=args.src_file, dest_file=args.dest_file)

    print("\n\nScript execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
