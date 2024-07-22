"""
Created 17 April 2024
@author: Dimitris Lymperopoulos
Description: A script containing a counterfactual editor class that uses a bipartite graph and a pretrained GNN to
create edits
"""

import pandas as pd
from torch_geometric.data import Data, Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from utils.glan_functions import *
from utils.graph_functions import *
from utils.search_funcs import *

torch.set_grad_enabled(False)


class AdapTopKGraph(torch.nn.Module):
    def __init__(self, step):
        super(AdapTopKGraph, self).__init__()
        self.step = step

    @staticmethod
    def knn_idx(distance_matrix, k):
        _, index = distance_matrix.sort(dim=1)

        return index[:, :k]

    def build_graph(self, distance_matrix):
        row_size, col_size = distance_matrix.shape
        k = min(row_size, 10 + self.step * int(row_size / 10))

        idx_knn = self.knn_idx(distance_matrix, k).reshape(-1, 1)
        idx_row = torch.range(0, row_size-1).view(-1, 1).repeat(1, k).reshape(-1, 1)
        idx_row = idx_row.type(torch.int64)

        edge_index = torch.cat((idx_row, idx_knn+row_size), dim=1).type(torch.long)
        edge_attr = distance_matrix[idx_row, idx_knn]

        edge_attr = torch.cat((edge_attr, edge_attr), dim=1).view(-1, 1)
        edge_index = torch.cat((edge_index, edge_index[:, 1].unsqueeze(1), edge_index[:, 0].unsqueeze(1)), dim=1)
        edge_index = edge_index.view(-1, 2).permute(1, 0)

        return edge_index, edge_attr, idx_row, idx_knn, k

    def forward(self, distance_matrix):
        edge_index, edge_attr, idx_row, idx_knn, k = self.build_graph(distance_matrix)

        if torch.cuda.is_available():
            data = Data(x=torch.zeros((sum(distance_matrix.shape), 8)).cuda(), edge_index=edge_index.cuda(),
                        edge_attr=edge_attr.cuda(),
                        kwargs=[distance_matrix.shape[0], distance_matrix.shape[1], k, idx_row, idx_knn,
                                edge_attr.shape[0]],
                        cost_vec=distance_matrix.view(-1, 1).cuda())
        else:
            data = Data(x=torch.zeros((sum(distance_matrix.shape), 8)), edge_index=edge_index,
                        edge_attr=edge_attr,
                        kwargs=[distance_matrix.shape[0], distance_matrix.shape[1], k, idx_row, idx_knn,
                                edge_attr.shape[0]],
                        cost_vec=distance_matrix.view(-1, 1))

        return data


class GraphData(Dataset):
    def __init__(self, data_matrix, step=2):
        super(GraphData, self).__init__()
        '''Initialization'''

        transor = AdapTopKGraph(step)
        self.data_item = transor(data_matrix)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data_item


class GnnEditor:
    def __init__(self, data, gnn_model, predictor=None, tokenizer=None, pos=None, antonyms=None, subs=None,
                 word_embeddings=None):
        """
        Initialize the GnnEditor object.

        :param data: dataframe with one column containing textual data
        :param gnn_model: GNN model to be used for solving RLAP problem
        :param predictor: pretrained model to be used for prediction
        :param tokenizer: Tokenizer to be used for tokenization before prediction
        :param pos: string representing Part-of-Speech (pos) tag
        :param antonyms: whether to substitute with antonyms or not
        :param subs: dictionary containing precomputed substitution pairs
        :param word_embeddings: dictionary containing word embeddings
        """
        self.data = data
        self.model = gnn_model
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.pos = pos
        self.antonyms = False if antonyms is None else antonyms

        self.fluency_model, self.fluency_tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())

        self.distance_matrix = None
        self.d0 = None
        self.d1 = None
        self.all_syn0 = None
        self.all_syn1 = None

        self.substitutions = subs
        self.word_embeddings = word_embeddings

    def create_distance_matrix(self, edge_filter=False):
        """
        Create a graph from the given data.
        :param edge_filter: boolean value, denoting whether to use edge filtering or not

        :return: GnnEditor object
        """
        sentences = [elem[0] for elem in self.data.values.tolist()]

        # use appropriate function based on pos to get the list of the specified pos words from the data
        if self.pos is not None:
            if self.pos == 'adj':
                lst = create_attributes_list(sentences)
            elif self.pos == 'verb':
                lst = create_verb_list(sentences)
            elif self.pos == 'noun':
                lst = create_singular_list(sentences)
            else:
                raise AttributeError("pos '{}' is not supported!".format(self.pos))

            syn0 = list(lst)
            syn1 = get_antonym_list(lst) if self.antonyms else syn0

        else:
            syn0 = []
            syn0.extend(create_attributes_list(sentences))
            syn0.extend(create_verb_list(sentences))
            syn0.extend(create_singular_list(sentences))

            syn1 = get_antonym_list(syn0) if self.antonyms else syn0

        self.all_syn0, self.d0, ind0 = get_synsets(syn0, pos=self.pos, return_index=True)
        self.all_syn1, self.d1, ind1 = get_synsets(syn1, pos=self.pos, return_index=True)

        # TODO: maybe use the min list as rows and the max as columns instead
        row_length = len(self.all_syn0)
        col_length = len(self.all_syn1)

        print("Creating Distance Matrix...")
        self.distance_matrix = torch.zeros((row_length, col_length))

        if self.word_embeddings is None:
            model_id = "llmrails/ember-v1"
            embed_tokenizer = AutoTokenizer.from_pretrained(model_id)
            embed_model = AutoModel.from_pretrained(model_id)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embed_model.to(device)

            for i in range(row_length):
                for j in range(col_length):
                    # rows will be syn0 and columns will be syn1
                    self.distance_matrix[i, j] = get_cos_similarity(
                        self.all_syn0[i], self.all_syn1[j], self.d0[self.all_syn0[i]],
                        self.d1[self.all_syn1[j]], embed_model, embed_tokenizer) if edge_filter else\
                        wn_path_similarity(self.all_syn0[i], self.all_syn1[j])

            del embed_model
            del embed_tokenizer

        else:
            # if embeddings are already precomputed
            for i in range(row_length):
                for j in range(col_length):
                    # get the word embeddings
                    try:
                        i_word_vector = self.word_embeddings[self.d0[self.all_syn0[i]]]
                        j_word_vector = self.word_embeddings[self.d1[self.all_syn1[j]]]
                    except KeyError:
                        self.distance_matrix[i, j] = 10
                        continue

                    if edge_filter and self.all_syn0[i].pos() != self.all_syn1[j].pos():
                        self.distance_matrix[i, j] = 10
                    else:
                        self.distance_matrix[i, j] = cosine(i_word_vector, j_word_vector)

        return self

    def find_substitutions(self):
        """
        Find the optimal substitutions from the distance matrix, using GLAN to solve the RLAP problem of
        the given distance matrix.

        :return: GnnEditor object
        """

        print("Finding Substitutions...")

        # create dataset and dataloader from the distance matrix
        dataset = GraphData(data_matrix=self.distance_matrix)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # get the minimum match of the distance matrix from the model output
        pred_matrix = None
        for batch_idx, cur_data in enumerate(dataloader):
            dt_target = cur_data
            shapes_info = dt_target.kwargs[0]

            # get model output
            shape1, shape2, k, idx_row, idx_knn, num_edges = shapes_info
            pred = self.model(dt_target)
            pred = pred.view(-1, 1)

            # convert model output to a matrix denoting whether an edge is part of the minimum match or not
            tag_scores = torch.zeros((1, shape1, shape2)).cuda()
            tag_scores[:, idx_row, idx_knn] = pred
            tag_scores = tag_scores.squeeze(0)

            pred_matrix = tag_scores.data.cpu().numpy()
            pred_matrix = sinkhorn_v1_np(pred_matrix + 1e-9)
            pred_matrix = greedy_map(pred_matrix)

        for i in range(pred_matrix.shape[0]):
            for j in range(pred_matrix.shape[1]):
                if pred_matrix[i][j] == 1:
                    self.substitutions[self.d0[self.all_syn0[i]]] = self.d1[self.all_syn1[j]]

        return self

    def create_counterfactuals(self, opt_th, use_contrastive_prob):
        """
        Create counterfactuals for the given data using the substitutions found.

        :param opt_th: whether to use optimal threshold for the number of substitutions
        :param use_contrastive_prob: whether to use contrastive probability as beam search criterion
        :return: List of the generated counterfactuals
        """

        print("Creating Counterfactuals...")

        sentences = [elem[0] for elem in self.data.values.tolist()]
        counter_sents = []
        for s in tqdm(sentences):
            # get original prediction probabilities
            logits = get_prediction(model=self.predictor, tokenizer=self.tokenizer, text=s, return_logits=True)
            probs = torch.softmax(logits, dim=1)[0]

            # get original fluency
            fluency = sent_scoring(self.fluency_model, self.fluency_tokenizer, s)[0]

            # get the best counterfactual using beam search
            max_subs = math.ceil(len(s.split()) / 5) if opt_th else 10
            cs = beam_search(text=s, substitutions=self.substitutions, original_probs=probs, original_fluency=fluency,
                             model=self.predictor, tokenizer=self.tokenizer, fluency_model=self.fluency_model,
                             fluency_tokenizer=self.fluency_tokenizer, max_subs=max_subs,
                             use_contrastive_prob=use_contrastive_prob)
            counter_sents.append(cs)

        counter_data = pd.DataFrame({
            'counter_sents': counter_sents
        })

        return counter_data, self.substitutions

    def pipeline(self, edge_filter=False, opt_th=False, use_contrastive_prob=False):
        """
        A method that provides sequential execution of the other methods to perform counterfactual generation.

        :param edge_filter: boolean value, denoting whether to apply edge filtering or not
        :param opt_th: boolean value, denoting whether to use optimal_threshold as upper limit of substitutions
        :param use_contrastive_prob: boolean value, denoting whether to include contrastive probability in beam_search
        :return:
        """

        if self.substitutions is None:
            self.substitutions = dict()
            return self.create_distance_matrix(edge_filter=edge_filter).find_substitutions().create_counterfactuals(
                opt_th=opt_th, use_contrastive_prob=use_contrastive_prob)
        else:
            return self.create_counterfactuals(opt_th=opt_th, use_contrastive_prob=use_contrastive_prob)
