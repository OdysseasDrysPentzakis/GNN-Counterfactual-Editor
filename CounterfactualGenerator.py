import argparse
import os
import pandas as pd
from datetime import datetime
from Editors.DummyEditor import DummyEditor


class CounterfactualGenerator:
    def __init__(self, src_file=None, src_sentence=None, indicative_sentence=None, dest_file=None, separator=None,
                 pos=None, synonyms=False):
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
        self.editor = DummyEditor(pos=pos, synonyms=synonyms)

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
