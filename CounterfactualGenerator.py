"""
Created 10 May 2023
@author: Dimitris Lymperopoulos
Description: A script that generates dummy counterfactual explanations for a given sentence or sentences

Usage:
1)  Generate a counterfactual sentence for a single given source_sentence, using synonyms as replacements
    python3 DummyGenerator.py
        --src-sentence <a string representing the original sentence>
        --indicative-sentence < a string indicating which words should be changed in the original sentence>
        --synonyms
        [--pos  <['n', 'a', 'v', 'r', 's'], represents part-of-speech of words that will be changed>]

2)  Generate counterfactual sentences based on a given source file and store the results in a destination file
    python3 DummyGenerator.py
        --src-file <.csv file containing two columns: Source_Sentences,Indicative_Sentences>
        [--dest-file < .csv file in which the generated sentences will be stored - default is src_file>]
        [--sep < a string representing the separator of the src_file - default is comma(',')>]

Example:
1)
    python3 DummyGenerator.py
        --src-sentence "A beautiful movie with great plot and interesting characters!"
        --indicative-sentence "A [BLANK] movie with [BLANK] plot and [BLANK] characters!"
        --synonyms

2)
    python3 DummyGenerator.py
        --src-file ~/sentences.csv
        --dest-file ~/generated_sentences.csv
        --sep "|"
        --pos "a"

"""

import argparse
import os
import spacy
import pandas as pd
from datetime import datetime
from Editors.DummyEditor import DummyEditor
from nltk import pos_tag, word_tokenize

nlp = spacy.load("en_core_web_sm")

nltk_pos_map = {
    'N': 'n',
    'V': 'v',
    'J': 'a'
}

spacy_pos_map = {
    'ADJ': 'a',
    'NOUN': 'n',
    'AUX': 'v',
    'VERB': 'v'
}


def create_indicative_sentence(s, pos, tagger='wordnet'):
    """
    :param s: A string representing the original sentence
    :param pos: part-of-speach of the words that need to be candidates for changing
    :param tagger: which of wordnet or spacy pos_taggers to use - default is wordnet
    :return: An indicative sentence in the polyjuice format, that dictates which words should be replaced
    """

    candidate_words = set()
    if tagger == 'wordnet':
        candidate_words = {
            word for (word, pos_) in pos_tag(word_tokenize(s)) if nltk_pos_map.get(pos_[0], pos_[0]) == pos
        }
    elif tagger == 'spacy':
        candidate_words = {str(word) for word in nlp(s) if spacy_pos_map.get(word.pos_, word.pos_) == pos}
    else:
        print("[ERROR]: Only legal options for 'tagger' parameter are wordnet and spacy!")
        exit(1)

    indicative_sentence = " ".join(list(map(lambda x: '[BLANK]' if x in candidate_words else x, s.split())))
    print(indicative_sentence)
    print(candidate_words)

    return indicative_sentence


class DummyGenerator:
    def __init__(self, src_file=None, src_sentence=None, indicative_sentence=None, dest_file=None, separator=None,
                 pos=None, synonyms=None):
        """
        :param src_file: a csv file containing Source_Sentences and Indicative_Sentences
        :param src_sentence: a string representing the source sentence that will be used to create a counterfactual
        :param indicative_sentence: a string that indicates which words shall be changed in the source_sentence
        :param dest_file: the destination csv file in which the generated sentences will be stored
        :param separator: the delimiter that separates the columns in the source_file
        :param pos: a string indicating the part-of-speech that will be targeted (n, v, a, r, etc.)
        :param synonyms: a boolean parameter indicating if synonyms or antonyms should be used as replacements
        """

        self.sentences = None
        self.src_sentence = None
        self.dest_file = None
        self.pos = pos
        self.separator = ',' if separator is None else separator
        self.synonyms = True if synonyms else False
        # check if source file or source sentence was given
        if src_file is None:
            if src_sentence is None:
                print("[ERROR]: At least one of source file, source sentence is required!")
                exit(1)
            # if src_sentence was given, indicative_sentence should also be given
            if indicative_sentence is None:
                print("[ERROR]: Indicative sentence is required!")
                exit(1)
            self.src_sentence = src_sentence
            self.indicative_sentence = indicative_sentence

        # if source file was given, check that it exists and then read it
        if src_file is not None and os.path.exists(src_file):
            # check that only one of src_file or src_sentence was given
            if src_sentence is not None:
                print("[ERROR]: Only one of source file, source sentence can be given!")
                exit(1)
            self.sentences = pd.read_csv(src_file, delimiter=self.separator)
            self.dest_file = src_file if dest_file is None else dest_file

            # if indicative sentences are not in the source file, then pos dictates what words to change
            if 'Indicative_Sentences' not in self.sentences.columns:
                if self.pos is None:
                    print("[ERROR]: Indicative Sentences and pos cannot be both None!")
                    exit(1)
                self.sentences['Indicative_Sentences'] = self.sentences['Source_Sentences'].apply(
                    lambda x: create_indicative_sentence(x, self.pos, tagger='spacy'))

        # editor that will perform the necessary changes to src_sentence or src_sentences
        self.editor = DummyEditor(pos=self.pos, synonyms=self.synonyms)

    def generate_counterfactuals(self, sentence):
        """
        A function that generates counterfactual sentences based on a sentence or a dataframe of sentences. In th first
        case, it prints the generated sentence, while in the second case it appends in the dataframe a column with the
        generated sentences

        :param sentence: a boolean parameter indicating whether to generate one counterfactual (True) or many (False)
        :return: a CounterfactualGenerator object
        """

        print("[INFO]: Generating counterfactuals...")

        if sentence:
            try:
                counterfactual = self.editor.generate_counterfactual(self.src_sentence, self.indicative_sentence,
                                                                     word_similarity='spacy')
                print("The following counterfactual was generated:")
                print(counterfactual)
            except IndexError:
                print("[ERROR]: Could not create a counterfactual for this sentence!")
                exit(1)

        else:
            generated_sentences = []
            for i in range(self.sentences.shape[0]):
                try:
                    new_sentence = self.editor.generate_counterfactual(self.sentences['Source_Sentences'][i],
                                                                       self.sentences['Indicative_Sentences'][i],
                                                                       word_similarity='wordnet'
                                                                       )
                    generated_sentences.append(new_sentence)
                except IndexError:
                    generated_sentences.append(self.sentences['Source_Sentences'][i])

            self.sentences['Counterfactual_Sentences'] = generated_sentences
            return self

    def export_to_csv(self):
        """
        A function that saves a dataframe with columns ['Source_Sentences', 'Indicative_Sentences',
        'Counterfactual_Sentences'] in a given destination file

        :return: a CounterfactualGenerator object
        """
        print("[INFO]: Exporting generated sentences to {}".format(self.dest_file))

        self.sentences.to_csv(self.dest_file, index=False, sep=self.separator)
        return self

    def pipeline(self):
        if self.src_sentence is not None:
            self.generate_counterfactuals(sentence=True)
        else:
            self.generate_counterfactuals(sentence=False).export_to_csv()


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-file", type=str, action="store", metavar="src_file",
                        required=False, help="The csv file with two columns - Source_Sentences, Indicative_Sentences")
    parser.add_argument("-d", "--dest-file", type=str, action="store", metavar="dest_file",
                        required=False, help="A csv file in which the generated counterfactuals will be stored")
    parser.add_argument("--src-sentence", type=str, action="store", metavar="src_sentence",
                        required=False, help="The source sentence that will be used to create a single counterfactual")
    parser.add_argument("--indicative-sentence", type=str, action="store", metavar="indicative_sentence",
                        required=False, help="The sentence that indicates which words should be changed in the source"
                                             "sentence")
    parser.add_argument("--sep", type=str, action="store", metavar="sep",
                        required=False, help="The delimiter that separates columns in the src_file")
    parser.add_argument("-p", "--pos", choices=['n', 'v', 'a', 'r', 's'], action='store', metavar="pos",
                        required=False, help="The part-of-speech that the replacements should be")
    parser.add_argument("--synonyms", action='store_true', required=False,
                        help="Whether synonyms or antonyms should be used as replacements")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    dg = DummyGenerator(src_file=args.src_file, dest_file=args.dest_file, src_sentence=args.src_sentence,
                        indicative_sentence=args.indicative_sentence, separator=args.sep, pos=args.pos,
                        synonyms=args.synonyms)

    dg.pipeline()

    print("\n\nScript execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
