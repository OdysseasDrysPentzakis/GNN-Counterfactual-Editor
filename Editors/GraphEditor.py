"""
Created 1 November 2023
@author: Dimitris Lymperopoulos
Description: A script containing a counterfactual editor class that uses a bipartite graph to create edits
"""

import warnings
from datetime import datetime
from utils.framework_functions import *
from utils.evaluation_metrics import *

warnings.filterwarnings("ignore", category=FutureWarning)


class GraphEditor:
    def __init__(self, data=None, pos=None, antonyms=None, eval_metric=None, baseline_metric=None, maximize=None,
                 max_iter=None, thresh=None, debug=False):
        if data is None:
            print("[ERROR]: No data provided to the editor!")
            exit(1)
        self.data = data

        self.pos = pos

        if eval_metric is None:
            print("[ERROR]: No evaluation metric provided to the editor!")
            exit(1)
        self.eval_metric = eval_metric

        self.antonyms = antonyms if antonyms is not None else False

        self.baseline_metric = baseline_metric
        self.maximize = maximize if maximize is not None else False

        self.max_iter = max_iter if max_iter is not None else 100
        self.thresh = thresh if thresh is not None else 0.005

        self.graph_dict = None

        self.debug = debug

    def create_graph(self):
        """
        A method that creates a bipartite graph from the data provided.

        :return: GraphEditor object
        """

        if self.debug:
            print("[INFO]: Creating graph...")

        self.graph_dict = create_graph(self.data, self.pos, self.antonyms)

        return self

    def train_graph(self):
        """
        A method that trains the bipartite graph created by the create_graph() method.

        :return: GraphEditor object
        """

        if self.debug:
            print("[INFO]: Training graph...")

        self.graph_dict = train_graph(self.graph_dict, data=self.data, pos=self.pos, eval_metric=self.eval_metric,
                                      baseline_metric=self.baseline_metric, maximize=self.maximize,
                                      max_iterations=self.max_iter, th=self.thresh)

        return self

    def generate_counterfactuals(self):
        """
        A method that generates counterfactuals for the data provided.

        :return: a dataframe with the generated counterfactual data
        """
        if self.debug:
            print("[INFO]: Generating counterfactuals...")

        counter_data, selected_edges, subs = generate_counterfactuals(self.graph_dict, self.data, self.pos)

        return counter_data

    def pipeline(self):
        """
        A method that runs the pipeline of the editor.

        :return: a dataframe with the generated counterfactual data
        """

        return self.create_graph().train_graph().generate_counterfactuals()


if __name__ == '__main__':

    # parameters initialization
    sents = pd.read_csv("../Data/NEWSGROUPS/test/newsgroups_test.csv")[["text"]].head(10)
    POS = 'adv'
    ANTONYMS = True
    baseline = 0.478

    start = datetime.now()

    editor = GraphEditor(data=sents, pos=POS, antonyms=ANTONYMS, eval_metric='fluency',
                         baseline_metric=baseline, debug=True)

    generated_edits = editor.pipeline()

    model, tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())
    print("Fluency: {}\n".format(get_fluency(sents, generated_edits, model, tokenizer)))

    for pair in zip(sents.values.tolist(), generated_edits.values.tolist()):
        print("Original: {}\n\nCounter: {}".format(pair[0][0], pair[1][0]))
        print("=" * 100)

    print("Script execution time: {}".format(datetime.now() - start))
