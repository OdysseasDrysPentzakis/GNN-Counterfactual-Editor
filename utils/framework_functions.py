from utils.evaluation_metrics import *
from evaluate import load


def update_edges(edges, substitutions, lr, baseline_metric_value, current_metric_value):
    """
    A function that takes as input a list of weighted edges along with other parameters, and uses
    these parameters to update the edge weights.

    :param edges: an iterable containing weighted edges as tuples
    :param substitutions: a dictionary with edges as keys, and their substitution occurence as values
    :param lr: float value, representing the learning rate for the weight updating
    :param baseline_metric_value: a float, representing the baseline evaluation metric value
    :param current_metric_value: a float, representing the current evaluation metric value
    :returns: a list of tuples, where each tuple represents an updated weighted edge
    """

    updated_edges = list()
    for (u, v, w) in edges:
        try:
            # get substitution occurences for each edge
            edge_subs = substitutions[(u, v)]
            # updating formula: Use - for minimizing, + for maximizing
            new_w = w - lr * (baseline_metric_value - current_metric_value) / edge_subs
            # add the updated edge to the list
            updated_edges.append((u, v, new_w))
        except KeyError:
            print("Something went wrong during updating of edges' weights")

    return updated_edges


def create_graph(data, pos, antonyms=False):
    """
    A function that takes as input a dataframe and a part-of-speech tag, and creates a bipartite graph
    with the possible substitution words and their candidates.

    :param data: pd.DataFrame() containing one column with the textual data
    :param pos: string that specifies which part-of-speech shall be considered for substitution (noun, verb, adv)
    :param antonyms: boolean value specifying whether to use antonyms in the candidate substitutions
    :returns: a dictionary containing the graph, along with other related features
    """

    sentences = [elem[0] for elem in data.values.tolist()]

    # use appropriate function based on pos to get the list of the specified pos words from the data
    if pos == 'adv':
        lst = create_attributes_list(sentences)
    elif pos == 'verb':
        lst = create_verb_list(sentences)
    elif pos == 'noun':
        lst = create_singular_list(sentences)
    else:
        raise AttributeError("pos '{}' is not supported!".format(pos))

    syn0 = list(lst)
    syn1 = list(get_antonym_list(lst)) if antonyms else list(lst)

    all_syn0, d0, ind0 = get_synsets(syn0, pos=pos, return_index=True)
    all_syn1, d1, ind1 = get_synsets(syn1, pos=pos, return_index=True)

    print("Creating Node Names...")
    names0 = ['G0_' + str(i) for i in range(len(all_syn0))]  # give unique names for each synset of the two sets
    names1 = ['G1_' + str(i) for i in range(len(all_syn1))]

    word_to_node0 = dict()
    word_to_node1 = dict()
    for t in zip(names0, ind0):
        word_to_node0[syn0[t[1]]] = t[0]

    for t in zip(names1, ind1):
        word_to_node1[syn1[t[1]]] = t[0]

    # synset as key, word as val
    combinations_nodes = all_combinations(names0, names1)  # all combinations of names
    combinations_synsets = all_combinations(all_syn0, all_syn1)  # all combinations of synsets
    weights = [1] * len(combinations_nodes)

    print("Creating Bipartite Graph...")
    g, min_list_nodes = bipartite_graph(names0, names1, combinations_nodes, weights)  # create bipartite graph

    graph_dict = {
        'graph': g,
        'min_list_nodes': min_list_nodes,
        'weights': weights,
        'd0': d0,
        'd1': d1,
        'comb_nodes': combinations_nodes,
        'comb_syn': combinations_synsets,
        'word_to_node0': word_to_node0,
        'word_to_node1': word_to_node1
    }

    return graph_dict


def generate_graph_matching(graph_dict):
    """
    A function that takes as input a dictionary containing a graph and other related features, and uses
    a minimum graph matching algorithm to return candidate substitutions, along with other graph features.

    :param graph_dict: a dictionary containing a bipartite graph and other related features
    :returns: a list of feasible substitutions, mappings of synsets to their words, and a tuple containing the graph,
    a min_list_nodes and the minimum matching
    """

    # unpack dictionary items
    g = graph_dict['graph']
    min_list_nodes = graph_dict['min_list_nodes']
    # weights = graph_dict['weights']
    d0 = graph_dict['d0']
    d1 = graph_dict['d1']
    combinations_nodes = graph_dict['comb_nodes']
    combinations_synsets = graph_dict['comb_syn']

    # find min weight match
    print("Finding Minimum Match...")
    min_match = minimum_match(g, min_list_nodes)
    match_tuple = dict_to_tuple(min_match)

    new_match = []
    for i in match_tuple:
        new_match.append(tuple(sorted(i)))
        new_match = remove_duplicates(new_match)

    positions = pos_in_list(combinations_nodes, list(new_match))
    substitution_synsets = dict()

    print("Creating Substitution Synsets Dictionary...")
    for i in positions:
        # substitution_synsets.append((weights[i], combinations_synsets[i][0], combinations_synsets[i][1]))
        substitution_synsets[d0[combinations_synsets[i][0]]] = d1[combinations_synsets[i][1]]
        substitution_synsets[d1[combinations_synsets[i][1]]] = d0[combinations_synsets[i][0]]

    return substitution_synsets, d0, d1, (g, min_list_nodes, new_match)


def generate_counterfactuals(graph_dict, data, pos):
    """
    A function that takes as input a dictionary containing graph information, along with a dataframe and a
    part-of-speech tag, and uses them to generate counterfactual edits from the data.

    :param graph_dict: a dictionary containing a bipartite graph and other related features
    :param data: pd.DataFrame() containing one column with the textual data
    :param pos: string that specifies which part-of-speech shall be considered for substitution (noun, verb, adv)
    :returns: a dataframe with the generated counterfactual data, a list of selected edges from the graph and a
    dictionary containing substitution occurrence
    """

    w2n0 = graph_dict['word_to_node0']
    w2n1 = graph_dict['word_to_node1']
    sentences = [elem[0] for elem in data.values.tolist()]

    # find best matching and generate edits
    substitution_synsets, d0, d1, g = generate_graph_matching(graph_dict)
    g = g[0]

    print("Generating Edits...")
    all_swaps, if_change, attr_counter, substitutions = external_swaps(sentences, pos, substitution_synsets, d0, d1,
                                                                       thresh=3)

    counter_data = pd.DataFrame({
        'counter_sents': all_swaps
    })

    subs_as_nodes = dict()
    for (k, v) in substitutions.items():
        try:
            subs_as_nodes[(w2n0[k[0]], w2n1[k[1]])] = v
        except KeyError:
            subs_as_nodes[(w2n0[k[1]], w2n1[k[0]])] = v

    selected_edges = []
    for (u, v) in subs_as_nodes.keys():
        w = g.get_edge_data(u, v, default=0)['weight']
        selected_edges.append((u, v, w))

    return counter_data, selected_edges, subs_as_nodes


def train_graph(graph_dict, data, pos, eval_metric, preprocessor=None, model=None, learning_rate=0.1, th=0.005,
                max_iterations=100, model_required=False, baseline_metric=None):
    """
    A function that represents the training process for the graph edges. It gets predictions for the original data
    then uses a graph approach to generate counter data and get predictions for them. To get the current_metric
    it compares the two predictions and based on those updates the weights of the selected edges.

    :param graph_dict: a dictionary containing the bipartite graph along with other variables and characteristics
    :param data: a dataframe containing the textual examples we will use to train the graph
    :param pos: a string specifing which part-of-speech shall be considered for substitutions (noun, verb, adv)
    :param eval_metric: a string that represents the metric which must be optimized during fine-tuning
    :param preprocessor: a custom class that implements the necessary preprocessing of the data
    :param model: a pretrained model on the dataset
    :param learning_rate: float value defining how fast or slow the edge weights will be updated
    :param th: float value defining a threshold, where if the difference |baseline - current| get smaller, the training
    stops
    :param max_iterations: integer value representing the maximum number of iterations for the training procedure
    :param model_required: boolean value for whether or not to compute model-related metrics
    :param baseline_metric: float value representing the baseline metric value
    :returns: the graph_dictionary with the fine-tuned (post-training) graph along with the rest of its features
    """

    # initialize baseline and current metric so that the dif |baseline-current| is bigger than th
    if baseline_metric is None:
        baseline_metric = get_baseline_metric(data, pos=pos, eval_metric=eval_metric, model_required=model_required,
                                              preprocessor=preprocessor, model=model)[0]
    current_metric = baseline_metric + 2 * th   # use - for metric that needs to be maximized

    original_preds = None
    if model_required:
        original_preds = model.predict(preprocessor.process(data))

    iterations = 0
    next_baseline_metric = baseline_metric
    fluency_model, fluency_tokenizer, bertscore = None, None, None

    if eval_metric == 'fluency':
        fluency_model, fluency_tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())
    elif eval_metric == 'bertscore':
        bertscore = load("bertscore")
    elif eval_metric == 'fluency_bertscore':
        model, tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())
        bertscore = load("bertscore")
    elif eval_metric == 'closeness':
        pass
    else:
        raise AttributeError("eval_metric '{}' is not supported!".format(eval_metric))

    while abs(current_metric - baseline_metric) >= th and iterations < max_iterations:
        print("ITERATION {}".format(iterations))

        # updated_edges = []
        baseline_metric = next_baseline_metric

        counter_data, selected_edges, substitutions = generate_counterfactuals(graph_dict, data, pos)

        print("Evaluating edits...")

        if not model_required:
            # compute current_metric valule
            if eval_metric == 'fluency':
                current_metric = get_fluency(data, counter_data, fluency_model, fluency_tokenizer)
            elif eval_metric == 'bertscore':
                current_metric = get_bertscore(data, counter_data, bertscore)
            elif eval_metric == 'fluency_bertscore':
                fluency = get_fluency(data, counter_data, fluency_model, fluency_tokenizer)
                bscore = get_bertscore(data, counter_data, bertscore)
                current_metric = 2 * fluency * bscore / (fluency + bscore)
            elif eval_metric == 'closeness':
                current_metric = get_closeness(data, counter_data)
            else:
                pass

        else:
            processed_counter_data = preprocessor.process(counter_data)
            counter_preds = model.predict(processed_counter_data)

            # compute model-related current_metric value
            current_metric = eval_metric(original_preds, counter_preds)

        # for sample information
        if iterations % 5 == 0:
            print(f"Current metric value: {current_metric}")

        g = graph_dict['graph']
        g.remove_edges_from(selected_edges)
        new_edges = update_edges(selected_edges, substitutions, learning_rate, baseline_metric, current_metric)
        g.add_weighted_edges_from(new_edges)
        graph_dict['graph'] = g

        ##############################################################################################################
        # while nx.is_bipartite(graph_dict['graph']):
        #     try:
        #         counter_data, selected_edges, substitutions = generate_counterfactuals(graph_dict, data, pos)
        #
        #         if not model_required:
        #             # compute current_metric valule
        #             current_metric = eval_metric(data, counter_data)
        #
        #         else:
        #             processed_counter_data = preprocessor.process(counter_data)
        #             counter_preds = model.predict(processed_counter_data)
        #
        #             # compute model-related current_metric value
        #             current_metric = eval_metric(original_preds, counter_preds)
        #
        #         g = graph_dict['graph']
        #         g.remove_edges_from(selected_edges)
        #         new_edges = update_edges(selected_edges, substitutions, learning_rate, baseline_metric,
        #         current_metric)
        #
        #         graph_dict['graph'] = g
        #         updated_edges.extend(new_edges)
        #     except:
        #         graph_dict['graph'] = g
        #         break
        #
        # g = graph_dict['graph']
        # g.add_weighted_edges_from(updated_edges)
        # graph_dict['graph'] = g
        ##############################################################################################################

        # update baseline_metric value and iterations
        next_baseline_metric = min(baseline_metric, current_metric)   # use max for metric that needs to be maximized
        iterations += 1

    return graph_dict
