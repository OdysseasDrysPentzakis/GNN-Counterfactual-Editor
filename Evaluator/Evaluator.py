"""
Created 30 October 2023
@author: Dimitris Lymperopoulos
Description: A script that evaluates given counterfactuals based on a specified metric

Usage:
1)  Compute desired metric for the generated counterfactuals. If no metric is specified, all metrics will be computed:
    python3 Evaluator.py
        --original-csv <path_to_original_csv>
        --original-col <name of column with the original sentences>
        --counter-csv <path_to_counter_csv>
        --counter-col <name of column with the counter sentences>
        [--metric <metric_to_be_used (fluency, bertscore, closeness, flip-rate)>]

Example:
1)
    python3 Evaluator.py
        --original-csv ~/data/original_data.csv
        --original-col sentences
        --counter-csv ~/data/counterfactual_data.csv
        --counter-col counter_sents
        --metric bertscore

"""

import os
import argparse
import torch.cuda
import warnings
from datetime import datetime
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from utils.evaluation_metrics import *

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)


class Evaluator:
    def __init__(self, original_csv=None, original_col=None, counter_csv=None, counter_col=None, metric=None,
                 predictor=None):
        if original_csv is None:
            print("[ERROR]: original_csv must be specified")
            exit(1)
        if not os.path.exists(original_csv):
            print("[ERROR]: file {} does not exist".format(original_csv))
            exit(1)
        if original_col is None:
            print("[ERROR]: original_col must be specified")
            exit(1)

        if counter_csv is None:
            print("[ERROR]: counter_csv must be specified")
            exit(1)
        if not os.path.exists(original_csv):
            print("[ERROR]: file {} does not exist".format(counter_csv))
            exit(1)
        if counter_col is None:
            print("[ERROR]: counter_col must be specified")
            exit(1)

        self.sents = pd.read_csv(original_csv)[[original_col]]
        self.counter_sents = pd.read_csv(counter_csv)[[counter_col]]
        self.metric = metric if metric is not None else 'all'
        self.predictor = None
        self.tokenizer = None

        if self.metric == 'flip-rate' or self.metric == 'all':
            if predictor is None:
                print("[ERROR]: predictor must be specified for flip-rate metric")
                exit(1)
            if not os.path.exists(predictor):
                print("[ERROR]: file {} does not exist".format(predictor))
                exit(1)
            self.predictor = DistilBertForSequenceClassification.from_pretrained(predictor)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def evaluate(self):
        print("[INFO]: Evaluating counterfactuals...")

        if self.metric == 'fluency':
            model, tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())
            fluency = get_fluency(self.sents, self.counter_sents, model, tokenizer)
            print("\nFluency: {}".format(fluency))

        elif self.metric == 'bertscore':
            bertscore = load("bertscore")
            score = get_bertscore(self.sents, self.counter_sents, bertscore)
            print("\nBERTScore: {}".format(score))

        elif self.metric == 'closeness':
            closeness = get_closeness(self.sents, self.counter_sents)
            print("\nCloseness: {}".format(closeness))

        elif self.metric == 'flip-rate':
            flip_rate = get_flip_rate(self.sents, self.counter_sents, self.predictor, self.tokenizer)
            print("\nFlip Rate: {}".format(flip_rate))

        elif self.metric == 'all':
            model, tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())
            fluency = get_fluency(self.sents, self.counter_sents, model, tokenizer)
            del model
            del tokenizer

            bertscore = load("bertscore")
            score = get_bertscore(self.sents, self.counter_sents, bertscore)

            closeness = get_closeness(self.sents, self.counter_sents)

            flip_rate = get_flip_rate(self.sents, self.counter_sents, self.predictor, self.tokenizer)

            print("\nFluency: {}".format(fluency))
            print("BERTScore: {}".format(score))
            print("Closeness: {}".format(closeness))
            print("Flip Rate: {}".format(flip_rate))


def parse_input(args=None):
    """
        param args: The command line arguments provided by the user
        :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--original-csv", action='store', metavar="original_csv", required=True,
                        help="The path to the csv file containing the original sentences")
    parser.add_argument("-oc", "--original-col", action='store', metavar="original_col", required=True,
                        help="The name of the column containing the original sentences")
    parser.add_argument("-c", "--counter-csv", action='store', metavar="counter_csv", required=True,
                        help="The path to the csv file containing the counterfactual sentences")
    parser.add_argument("-cc", "--counter-col", action='store', metavar="counter_col", required=True,
                        help="The name of the column containing the counterfactual sentences")
    parser.add_argument("-m", "--metric", choices=['fluency', 'bertscore', 'closeness', 'flip-rate'], action='store',
                        metavar="metric", required=False, default='all', help="The metric to be used for evaluation")
    parser.add_argument("-p", "--predictor", action='store', metavar="predictor", required=False,
                        help="The path to the pretrained predictor model")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    evaluator = Evaluator(original_csv=args.original_csv, original_col=args.original_col,
                          counter_csv=args.counter_csv, counter_col=args.counter_col, metric=args.metric,
                          predictor=args.predictor)

    evaluator.evaluate()

    print("\n\nScript execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
