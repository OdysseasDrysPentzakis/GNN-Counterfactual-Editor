import argparse
import os
import pandas as pd
from datetime import datetime
from Editors.DummyEditor import DummyEditor


class CounterfactualGenerator:
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
        self.separator = ',' if separator is None else separator
        self.synonyms = False if synonyms is None else True
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

        # editor that will perform the necessary changes to src_sentence or src_sentences
        self.editor = DummyEditor(pos=pos, synonyms=self.synonyms)

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
            counterfactual = self.editor.generate_counterfactual(self.src_sentence, self.indicative_sentence)
            print("The following counterfactual was generated:")
            print(counterfactual)

        else:
            generated_sentences = []
            for i in range(self.sentences.shape[1]):
                generated_sentences.append(self.editor.generate_counterfactual(self.sentences['Source_Sentences'][i],
                                                                               self.sentences['Indicative_Sentences'][i]
                                                                               ))
            self.sentences['Counterfactual_Sentences'] = generated_sentences
            return self

    def export_to_csv(self):
        """
        A function that saves a dataframe with columns ['Source_Sentences', 'Inidicative_Sentences',
        'Counterfactual_Sentences'] in a given destination file

        :return: a CounterfactualGenerator object
        """
        print("[INFO]: Exporting generated sentences to {}".format(self.dest_file))

        self.sentences.to_csv(index=False, sep=self.separator)
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
    parser.add_argument("--synonyms", action='store', metavar="synonyms", required=False,
                        help="Whether synononyms or antonyms should be used as replacements")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    cf_generator = CounterfactualGenerator(src_file=args.src_file, dest_file=args.dest_file,
                                           src_sentence=args.src_sentence, indicative_sentence=args.indicative_sentence,
                                           separator=args.sep, pos=args.pos, synonyms=args.synonyms)

    cf_generator.pipeline()

    print("\n\nScript execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
