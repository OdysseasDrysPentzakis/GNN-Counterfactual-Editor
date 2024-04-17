import os
import json
import argparse
from datetime import datetime
from GLAN_Model.GNBlock import Model
from Editors.GnnEditor import *
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


class GnnGenerator:
    def __init__(self, src_file=None, col=None, dest_file=None, json_file=None, pos=None, antonyms=None,
                 gnn_model_file=None, predictor_path=None):
        """
        A class that generates edits using a pretrained GNN model to solve RLAP.

        :param src_file: string(filepath) representing the source csv file containing the original sentences
        :param col: string representing the column name where the original sentences are stored
        :param dest_file: string(filepath) representing the destination csv file where the edits will be stored
        :param json_file: string representing the destination json file where the eligible substitutions will be stored
        :param pos: string representing the part-of-speech of words to be substituted
        :param antonyms: a boolean value specifying whether to use antonyms as substitutes
        :param gnn_model_file: string representing the .pth file where the pretrained GNN is stored
        :param predictor_path: string representing the directory where the pretrained classifier is stored
        """

        if src_file is None:
            print("[ERROR]: src_file must be specified")
            exit(1)
        if not os.path.exists(src_file):
            print("[ERROR]: File {} does not exist".format(src_file))
            exit(1)
        if col is None:
            print("[ERROR]: col must be specified")
            exit(1)
        self.sentences = pd.read_csv(src_file)[[col]].head(100)

        if gnn_model_file is None:
            print("[ERROR]: gnn_model_file must be specified")
            exit(1)
        if not os.path.exists(gnn_model_file):
            print("[ERROR]: File {} does not exist".format(gnn_model_file))
            exit(1)
        self.gnn_model = Model(layer_num=5, edge_dim=16, node_dim=8)
        self.gnn_model.load_state_dict(torch.load(gnn_model_file))

        if predictor_path is None:
            print("[ERROR]: predictor_file must be specified")
            exit(1)
        if not os.path.exists(predictor_path):
            print("[ERROR]: File {} does not exist".format(predictor_path))
            exit(1)
        self.predictor = DistilBertForSequenceClassification.from_pretrained(predictor_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        self.dest_file = 'gnn_edits.csv' if dest_file is None else dest_file
        self.json_file = json_file
        self.pos = pos
        self.antonyms = antonyms

        self.edits = None
        self.subs_dict = None

    def generate_counterfactuals(self):
        """
        A method that generates counterfactuals using a bipartite graph, a pretrained GNN model and beam search.

        :return: GnnGenerator object
        """
        print("[INFO]: Generating counterfactuals...")

        gnn_editor = GnnEditor(data=self.sentences, gnn_model=self.gnn_model, predictor=self.predictor,
                               tokenizer=self.tokenizer, pos=self.pos, antonyms=self.antonyms)
        self.edits, self.subs_dict = gnn_editor.pipeline()

        return self

    def export_to_csv(self):
        """
        A method that exports the generated counterfactuals to a csv file.

        :return: GnnGenerator object
        """

        print("[INFO]: Exporting generated counterfactuals to {}...".format(self.dest_file))
        self.edits.to_csv(self.dest_file, index=False)

        if self.json_file is not None:
            print("[INFO]: Exporting substitution dictionary to {}...".format(self.json_file))
            with open(self.json_file, 'w') as f:
                json.dump(self.subs_dict, f)

        return self

    def pipeline(self):
        """
        A method that executes the pipeline for generating counterfactuals using a bipartite graph and a pretrained gnn
        model.

        :return: None
        """

        self.generate_counterfactuals().export_to_csv()


def parse_input(args=None):
    """
        :param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-file", action='store', metavar="src_file", required=True,
                        help="The path to the csv file with the original sentences")
    parser.add_argument("-c", "--col", action='store', metavar="col", required=True,
                        help="The name of the column containing the original sentences")
    parser.add_argument("-d", "--dest-file", action='store', metavar="dest_file", required=False,
                        help="The csv filepath where the generated edits will be stored")
    parser.add_argument("-j", "--json-file", action='store', metavar="json_file", required=False,
                        help="The json filepath where the substitution dictionary will be stored")
    parser.add_argument("-p", "--pos", action='store', metavar="pos", required=False,
                        help="The part-of-speech tag of the words to be substituted")
    parser.add_argument("--antonyms", action='store_true', required=False,
                        help="Whether to use antonyms as substitutes")
    parser.add_argument("-g", "--gnn-model-file", action='store', metavar="gnn_model_file", required=True,
                        help="The path to the pretrained GNN model")
    parser.add_argument("--predictor-path", action='store', metavar="predictor_path", required=True,
                        help="The path to the pretrained classifier")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    generator = GnnGenerator(src_file=args.src_file, col=args.col, dest_file=args.dest_file, json_file=args.json_file,
                             pos=args.pos, antonyms=args.antonyms, gnn_model_file=args.gnn_model_file,
                             predictor_path=args.predictor_path)

    generator.pipeline()

    print("\n\nScript execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
