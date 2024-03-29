"""
Created 3 November 2023
@author: Dimitris Lymperopoulos
Description: A script containing a counterfactual generator class that uses a bipartite graph to generate edits

Usage:
1) Generate counterfactuals and store them to the default location, which is ./graph_edits.csv, while also using
    default values for max_iterations (100) and threshold (0.005).The default part-of-speech tag is 'adv' and the
    default value for antonyms is False:
    python3 GraphGenerator.py
        --src-file <path_to_src_file>
        --col <name of column with the original sentences>
        --metric <metric to be used for evaluation (fluency, bertscore, closeness, fluency_bertscore)>

2)  Generate counterfactuals by selecting the words that satisfy a given pos tag, and substituting them with either
    their synonyms or antonyms, while also selecting them in a way that optimizes a given metric:
    python3 GraphGenerator.py
        --src-file <path_to_src_file>
        --col <name of column with the original sentences>
        --metric <metric to be used for evaluation (fluency, bertscore, closeness, fluency_bertscore)>
        [--dest-file <path_to_dest_file>]
        [--pos <part-of-speech tag of words to be substituted>]
        [--antonyms]
        [--baseline <baseline metric value (minimum value to achieve)>]
        [--max-iter <maximum number of iterations when training the bipartite graph>]
        [--thresh <threshold for convergence of the graph training process>]

2) Generate counterfactuals and store them to the default location, which is ./graph_edits.csv, while also using
    default values for max_iterations (100) and threshold (0.005).The default part-of-speech tag is 'adv' and the
    default value for antonyms is False:
    python3 GraphGenerator.py
        --src-file <path_to_src_file>
        --col <name of column with the original sentences>
        --metric <metric to be used for evaluation (fluency, bertscore, closeness, fluency_bertscore)>

Example:
1) Generate the most generic counterfactuals by giving only the required parameters and leaving the rest to default:
    python3 GraphGenerator.py
        --src-file ~/data/original_data.csv
        --col sentences
        --metric fluency_bertscore

2)  Specify every parameter to generate as specific counterfactuals as possible:
    python3 GraphGenerator.py
        --src-file ~/data/original_data.csv
        --col sentences
        --metric fluency_bertscore
        --dest-file ~/data/counterfactual_data.csv
        --pos adv
        --antonyms
        --baseline 0.5
        --max-iter 100
        --thresh 0.005
"""

import os
import argparse
import pandas as pd
from datetime import datetime

from Editors.GraphEditor import GraphEditor


class GraphGenerator:
    def __init__(self, src_file=None, col=None, dest_file=None, pos=None, antonyms=None, baseline=None, metric=None,
                 maximize=None, max_iter=None, thresh=None):
        """
        A class that generates counterfactuals using a bipartite graph.
        :param src_file: string(filepath) representing the source csv file containing the original sentences
        :param col: string representing the column name where the original sentences are stored
        :param dest_file: string(filepath) representing the destination csv file where the edits will be stored
        :param pos: string representing the part-of-speech of words to be substituted
        :param antonyms: a boolean value specifying whether to use antonyms as substitutes
        :param baseline: a float value specifying the baseline metric value (minimum value to achieve)
        :param metric: a string representing the metric that needs to be optimized
        :param max_iter: an integer value indicating max nu,ber of iterations when trainig the bipartite graph
        :param thresh: a float value indicating the threshold for convergence of the graph training process
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
        self.sentences = pd.read_csv(src_file)[[col]]

        self.pos = 'adv' if pos is None else pos
        self.antonyms = False if antonyms is None else antonyms
        self.baseline = float(baseline) if baseline is not None else None

        if metric is None:
            print("[ERROR]: metric must be specified")
            exit(1)
        if metric not in {'fluency', 'closeness', 'bertscore', 'fluency_bertscore'}:
            print("[ERROR]: metric must be one of {fluency, closeness, bertscore, fluency_bertscore}")
            exit(1)
        self.metric = metric

        self.maximize = maximize

        self.max_iter = 100 if max_iter is None else int(max_iter)
        self.thresh = 0.005 if thresh is None else float(thresh)
        self.dest_file = 'graph_edits.csv' if dest_file is None else dest_file
        self.edits = None

    def generate_counterfactuals(self):
        """
        A method that generates counterfactuals using a bipartite graph.

        :return: GraphGenerator object
        """

        print("[INFO]: Generating counterfactuals...")
        editor = GraphEditor(data=self.sentences, pos=self.pos, antonyms=self.antonyms, eval_metric=self.metric,
                             baseline_metric=self.baseline, maximize=self.maximize, max_iter=self.max_iter,
                             thresh=self.thresh)
        self.edits = editor.pipeline()

        return self

    def export_to_csv(self):
        """
        A method that exports the generated counterfactuals to a csv file.

        :return: GraphGenerator object
        """

        print("[INFO]: Exporting generated counterfactuals to {}...".format(self.dest_file))
        self.edits.to_csv(self.dest_file, index=False)

        return self

    def pipeline(self):
        """
        A method that executes the pipeline for generating counterfactuals using a bipartite graph.

        :return: None
        """

        self.generate_counterfactuals().export_to_csv()


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-file", action='store', metavar="src_file", required=True,
                        help="The path to the csv file with the original sentences")
    parser.add_argument("-c", "--col", action='store', metavar="col", required=True,
                        help="The name of the column containing the original sentences")
    parser.add_argument("-d", "--dest-file", action='store', metavar="dest_file", required=False,
                        help="The csv filepath where the generated edits will be stored")
    parser.add_argument("-p", "--pos", action='store', metavar="pos", required=False,
                        help="The part-of-speech tag of the words to be substituted")
    parser.add_argument("--antonyms", action='store_true', required=False,
                        help="Whether to use antonyms as substitutes")
    parser.add_argument("-b", "--baseline", action='store', metavar="baseline", required=False,
                        help="The baseline metric value (minimum value to achieve)")
    parser.add_argument("-m", "--metric", choices=['fluency', 'bertscore', 'closeness', 'fluency_bertscore'],
                        action='store', metavar="metric", required=True,
                        help="The metric to be used for evaluation")
    parser.add_argument("--maximize", action='store_true', required=False,
                        help="whether to maximize or minimize the evaluation metric")
    parser.add_argument("--max-iter", action='store', metavar="max_iter", required=False,
                        help="The maximum number of iterations when training the bipartite graph")
    parser.add_argument("--thresh", action='store', metavar="thresh", required=False,
                        help="The threshold for convergence of the graph training process")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    generator = GraphGenerator(src_file=args.src_file, col=args.col, dest_file=args.dest_file, pos=args.pos,
                               antonyms=args.antonyms, baseline=args.baseline, metric=args.metric,
                               maximize=args.maximize, max_iter=args.max_iter, thresh=args.thresh)

    generator.pipeline()

    print("\n\nScript execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
