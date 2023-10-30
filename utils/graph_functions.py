import spacy
import tqdm
import re
import itertools
import networkx as nx

from nltk.corpus import wordnet as wn
from operator import itemgetter
from tqdm import tqdm
from networkx.algorithms import bipartite

nlp = spacy.load('en_core_web_sm')


def check_if_noun(sentence):
    singular = []
    plural = []
    # expressions containing nouns that should not be substituted
    exceptions = ['a group of', 'a couple of', 'group of', 'couple of', 'many',
                  'several', 'a lot of', 'lots of', 'others', 'other']
    expressions_to_pass = ['in front of', 'next to']
    txt = sentence.lower()
    for e in (exceptions+expressions_to_pass):
        txt = txt.replace(e, '')

    txt = ' '.join(txt.split())      # remove multiple whitespaces
    doc = nlp(txt)
    for token in doc:
        if len(token) > 1:
            if token.tag_ == 'NN':
                singular.append(token.text)
            elif token.tag_ == 'NNS':
                plural.append(token.text)
    return singular, plural, txt


def check_if_verb(txt):
    vbp = []
    vbg = []
    vb = []
    txt = txt.lower()
    txt = txt.strip()
    doc = nlp(txt)
    for token in doc:
        if token.dep_ != 'aux':         # except auxiliary verbs
            if token.tag_ == 'VBP':
                vbp.append(token.text)
            elif token.tag_ == 'VBG':
                vbg.append(token.text)
            elif token.tag_ == 'VB':
                vb.append(token.text)

    return vbp, vbg, vb, txt


def check_if_attribute(sentence):
    # expressions containing nouns that should not be substituted
    txt = sentence.lower()
    txt = ' '.join(txt.split())      # remove multiple whitespaces
    doc = nlp(txt)
    attr_list = []
    for token in doc:
        if token.pos_ == 'ADJ':
            attr_list.append(token.text)
    return attr_list, txt


def swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))


def change_verbs(caption, lst, change, nouns_changed):
    if len(lst) > 1:
        new_sent = swap_words(caption, lst[0], lst[1])
        nouns_changed.append(lst[0])
        nouns_changed.append(lst[1])
        change += 1
    else:
        new_sent = caption

    return new_sent, change, nouns_changed


def check_if_changed(val, lst):
    if val > 0:
        lst.append(1)
    else:
        lst.append(0)

    return lst


def ends_with_fullstop(txt):
    if txt.strip().endswith('.'):
        pass
    else:
        txt += '.'

    return txt


def list_diff(l1, l2):
    diff = [x for x in l1 if x not in set(l2)]
    return diff


def get_synsets(syn, pos, return_index=False):
    all_syn = []
    indices = []
    d = dict()
    p = pos[0]
    for idx, i in enumerate(syn):
        if wn.synsets(i, pos=p):
            s = wn.synsets(i, pos=p)[0]
            all_syn.append(s)
            d[s] = i
            indices.append(idx)
    if return_index:
        return all_syn, d, indices
    else:
        return all_syn, d


def get_antonym(given_word):
    antonyms = []
    for syn in wn.synsets(given_word):
        for lem in syn.lemmas():
            if lem.antonyms() and lem.antonyms()[0].synset().pos() == lem.synset().pos():
                antonyms.append(lem.antonyms()[0].name())
    return list(set(antonyms))   # remove duplicates and return the list


def get_antonym_list(words):
    ant_list = []
    for w in words:
        ant_list.extend(get_antonym(w))
    return ant_list


def all_combinations(a, b):   # give all combinations of two sets
    combinations = list(itertools.product(a, b))   # cartesian product
    return combinations


def shorter_list(l1, l2):
    min_list = min([l1, l2], key=len)  # find the shortest list between two lists
    return min_list


def swap(item):
    swaped = item[1], item[0]
    return swaped


def dict_to_tuple(d):
    tmp = d.items()   # get dict items
    lst = list(tmp)     # convert dict to list of tuples (k, v)
    return lst


def remove_duplicates(lst):
    lst = list(set(lst))
    return lst


def pos_in_list(lst, m):
    positions = []
    for i in m:
        swaped = swap(i)
        if i in lst:
            positions.append(lst.index(i))
        elif swaped in lst:                   # as an undirected graph, swaped edges are the same as in normal order
            positions.append(lst.index(swaped))
        else:
            print(i, 'not in list')
    return positions


def total_graph_weight(positions, weights, combinations_synsets):  # cumulative weight of bipartite matches
    sum_similarities = 0
    best_matched_synsets = []

    for i in positions:
        w = weights[i]
        sum_similarities += w
        synset_pair = combinations_synsets[i]
        best_matched_synsets.append([synset_pair[0], synset_pair[1], w])

    avg_similarity = sum_similarities / len(positions)

    return sum_similarities, avg_similarity, best_matched_synsets


def bipartite_graph(names0, names1, combinations_n, weights):
    g = nx.Graph()
    g0_nodes = names0
    g1_nodes = names1
    min_list_nodes = shorter_list(names0, names1)

    g.add_nodes_from(g0_nodes, bipartite=0)
    g.add_nodes_from(g1_nodes, bipartite=1)

    for name, w in zip(combinations_n, weights):
        g.add_edge(name[0], name[1], weight=w)

    if not nx.is_bipartite(g):
        print('Graph is not bipartite')

    return g, min_list_nodes


def minimum_match(g, min_list_nodes):
    min_matching = bipartite.matching.minimum_weight_full_matching(g, min_list_nodes, "weight")
    return min_matching


def wn_path_similarity(synset0, synset1):    # find wordnet path similarity score between two given synsets
    sim = synset0.path_similarity(synset1)
    return sim


def wn_hierarchy(s0, s1, pos, baseline=True):
    weights = []
    syn0 = list(s0)
    syn1 = list(s1)
    all_syn0, d0 = get_synsets(syn0, pos=pos)
    all_syn1, d1 = get_synsets(syn1, pos=pos)

    print("Creating Node Names...")
    names0 = ['G0_' + str(i) for i in range(len(all_syn0))]  # give unique names for each synset of the two sets
    names1 = ['G1_' + str(i) for i in range(len(all_syn1))]

    # synset as key, word as val
    combinations_nodes = all_combinations(names0, names1)  # all combinations of names
    combinations_synsets = all_combinations(all_syn0, all_syn1)  # all combinations of synsets

    if baseline:
        for comb, syn in zip(combinations_nodes, combinations_synsets):  # find path similarities for all combinations
            path_sim = wn_path_similarity(syn[0], syn[1])
            weights.append(path_sim)  # with those pairwise similarities acting as weights
    else:
        weights = [1] * len(combinations_nodes)

    print("Creating Bipartite Graph...")
    g, min_list_nodes = bipartite_graph(names0, names1, combinations_nodes, weights)  # create bipartite graph

    print("Finding Minimum Match...")
    min_match = minimum_match(g, min_list_nodes)  # find min weight match
    match_tuple = dict_to_tuple(min_match)

    new_match = []
    for i in match_tuple:
        # new_match.append(tuple(sorted(i)))
        # new_match = remove_duplicates(new_match)
        new_match.append(i)

    positions = pos_in_list(combinations_nodes, list(new_match))
    substitution_synsets = dict()

    print("Creating Substitution Synsets Dictionary...")
    for i in positions:
        # substitution_synsets.append((weights[i], combinations_synsets[i][0], combinations_synsets[i][1]))
        substitution_synsets[d0[combinations_synsets[i][0]]] = d1[combinations_synsets[i][1]]
        substitution_synsets[d1[combinations_synsets[i][1]]] = d0[combinations_synsets[i][0]]

    return substitution_synsets, d0, d1, (g, min_list_nodes, new_match)


def sort_select_triplet(lst):
    s_list = sorted(lst, key=itemgetter(0))   # sort list based on 1st element (weight). Smaller values come first
    selected = s_list[0]                      # get lower similarity triplet (first after sorting)
    return selected


def most_dissimilar_pair(substitution_synsets, d0, d1):
    selected = sort_select_triplet(substitution_synsets)  # selected format: (weight, synset0, synset1)
    p0 = d0[selected[1]]                                  # 1st synset at position 1
    p1 = d1[selected[2]]                                  # 2nd synset at position 2
    subs_pair = (p0, p1)
    return subs_pair


def create_attributes_list(sentences):
    all_attributes = []
    for s in sentences:
        attribute, new_s = check_if_attribute(s)
        all_attributes.append(attribute)

    attributes = [item for sublist in all_attributes for item in sublist]
    attributes = list(set(attributes))
    attributes = [word.replace('\\n', '') for word in attributes]
    attributes = [word.replace('\\', '') for word in attributes]

    return attributes


def graph_adverb_substitutions(sentences, pos, baseline=True, antonyms=False):
    """
    A function that takes as input a list od sentences, and generates substitutions.

    :param sentences: Iterable containing the sentences that will be changed
    :param pos: the part-of-speech of the words that will be changed
    :param baseline: boolean value specifying whether to use the baseline similarity function or not
    :param antonyms: boolean value specifying whether to use antonyms in the candidate substitutions
    :returns: the substitutions, the source set and the target set of the bipartite graph, the graph and the
    minimum_node_list
    """

    attributes = create_attributes_list(sentences)
    if antonyms:
        return wn_hierarchy(attributes, get_antonym_list(attributes), pos, baseline)
    else:
        return wn_hierarchy(attributes, attributes, pos, baseline)


def create_verb_list(sentences):
    all_vbp = []
    all_vbg = []
    all_vb = []

    for s in sentences:
        vbp, vbg, vb, new_s = check_if_verb(s)
        all_vbp.append(vbp)
        all_vbg.append(vbg)
        all_vb.append(vb)

    vbp = [item for sublist in all_vbp for item in sublist]
    vbp = list(set(vbp))
    vbp = [word.replace('\\n', '') for word in vbp]
    vbp = [word.replace('\\', '') for word in vbp]

    vbg = [item for sublist in all_vbg for item in sublist]
    vbg = list(set(vbg))
    vbg = [word.replace('\\n', '') for word in vbg]
    vbg = [word.replace('\\', '') for word in vbg]

    vb = [item for sublist in all_vb for item in sublist]
    vb = list(set(vb))
    vb = [word.replace('\\n', '') for word in vb]
    vb = [word.replace('\\', '') for word in vb]

    return vbp + vbg + vb


def graph_verb_substitutions(sentences, pos, baseline=True, antonyms=False):
    """
    A function that takes as input a list od sentences, and generates substitutions.

    :param sentences: Iterable containing the sentences that will be changed
    :param pos: the part-of-speech of the words that will be changed
    :param baseline: boolean value specifying whether to use the baseline similarity function or not
    :param antonyms: boolean value specifying whether to use antonyms in the candidate substitutions
    :returns: the substitutions, the source set and the target set of the bipartite graph, the graph and the
    minimum_node_list
    """

    verb_list = create_verb_list(sentences)
    if antonyms:
        return wn_hierarchy(verb_list, get_antonym_list(verb_list), pos, baseline)
    else:
        return wn_hierarchy(verb_list, verb_list, pos, baseline)


def create_singular_list(sentences):
    all_singulars = []
    # all_plurals = []

    for s in sentences:
        singular, plural, new_s = check_if_noun(s)
        all_singulars.append(singular)
        # all_plurals.append(plural)

    singulars = [item for sublist in all_singulars for item in sublist]
    singulars = list(set(singulars))
    singulars = [word.replace('\\n', '') for word in singulars]
    singulars = [word.replace('\\', '') for word in singulars]

    # all_plurals = [item for sublist in all_plurals for item in sublist]
    # plurals = list(set(all_plurals))
    # plurals = [word.replace('\\n', '') for word in plurals]
    # plurals = [word.replace('\\', '') for word in plurals]

    return singulars


def graph_noun_substitutions(sentences, pos, baseline=True, antonyms=False):
    """
    A function that takes as input a list od sentences, and generates substitutions.

    :param sentences: Iterable containing the sentences that will be changed
    :param pos: the part-of-speech of the words that will be changed
    :param baseline: boolean value specifying whether to use the baseline similarity function or not
    :param antonyms: boolean value specifying whether to use antonyms in the candidate substitutions
    :returns: the substitutions, the source set and the target set of the bipartite graph, the graph and the
    minimum_node_list
    """

    singulars = create_singular_list(sentences)
    if antonyms:
        return wn_hierarchy(singulars, get_antonym_list(singulars), pos, baseline)
    else:
        return wn_hierarchy(singulars, singulars, pos, baseline)


def external_swaps(sentences, pos, substitution_singular, d0_s, d1_s, thresh=100):
    """
    A function that takes as input a dataframe and the name of the column where the sentences are,
    and generates substitutions.

    :param sentences: Iterable containing the sentences that will be changed
    :param pos: a string specifing which part-of-speech shall be changed (attr, verb or noun)
    :param substitution_singular: an iterable with the possible substitutions
    :param d0_s: a list containing the nodes (words) of the source set of the bipartite graph
    :param d1_s: a list containing the nodes (words) of the target set of the bipartite graph
    :param thresh: Integer representing how many substitutions in each sentence shall occurr
    :returns: a triplet containing the swaps that were made, a list denoting which sentences were changed and how
    many attributes were changed
    """
    all_swaps = []
    if_change = []
    substitutions = dict()
    attr_counter = 0

    for s in tqdm(sentences):
        change = 0
        txt = s.lower().replace('\\n', '')

        # according to the pos given, use the appropriate function to create the list with candidate words to be
        # substituted
        if pos == 'adv':
            candidate_list, new_s = check_if_attribute(s)
        elif pos == 'verb':
            vbp, vbg, vb, new_c = check_if_verb(txt)
            candidate_list = vbp + vbg + vb
        elif pos == 'noun':
            candidate_list, plural, new_c = check_if_noun(txt)
        else:
            raise AttributeError("pos '{}' is not supported!".format(pos))

        # crop candidate list if it is larger than the threshold given
        if len(candidate_list) > thresh:
            candidate_list = candidate_list[:thresh]

        # perform substitutions
        # for i in substitution_singular:
        #     if d0_s[i[1]] in txt and (d0_s[i[1]] in candidate_list):
        #         txt=re.sub(r"\b%s\b" % d0_s[i[1]] , d1_s[i[2]], txt)
        #         change+=1
        #         attr_counter+=1
        #         substitutions[(d0_s[i[1]], d1_s[i[2]])] = substitutions.get((d0_s[i[1]], d1_s[i[2]]), 0) + 1

        #     elif d1_s[i[2]] in txt and (d1_s[i[2]] in candidate_list):
        #         txt = txt.replace(d1_s[i[2]], d0_s[i[1]])
        #         txt=re.sub(r"\b%s\b" % d1_s[i[2]] , d0_s[i[1]], txt)
        #         change+=1
        #         attr_counter+=1
        #         #substitutions[(d1_s[i[1]], d0_s[i[2]])] = substitutions.get((d1_s[i[1]], d0_s[i[2]]), 0) + 1
        #         substitutions[(d0_s[i[1]], d1_s[i[2]])] = substitutions.get((d0_s[i[1]], d1_s[i[2]]), 0) + 1

        for c in candidate_list:
            sub_word = substitution_singular.get(c, c)
            if c != sub_word:
                txt = re.sub(r"\b%s\b" % c, sub_word, txt)
                change += 1
                attr_counter += 1
                substitutions[(c, sub_word)] = substitutions.get((c, sub_word), 0) + 1

        # update related variables accordingly
        new_txt = ends_with_fullstop(txt)
        all_swaps.append(new_txt)
        if_change = check_if_changed(change, if_change)

    return all_swaps, if_change, attr_counter, substitutions


def get_edits(sentences, pos, thresh=100, baseline=True, antonyms=False):

    # use appropriate function based on pos to get feasible substitutions
    if pos == 'adv':
        substitution_singular, d0_s, d1_s, g = graph_adverb_substitutions(sentences, pos, baseline, antonyms)
    elif pos == 'verb':
        substitution_singular, d0_s, d1_s, g = graph_verb_substitutions(sentences, pos, baseline, antonyms)
    elif pos == 'noun':
        substitution_singular, d0_s, d1_s, g = graph_noun_substitutions(sentences, pos, baseline, antonyms)
    else:
        raise AttributeError("pos '{}' is not supported!".format(pos))

    # return the edited sentences
    print("Generating Edits...")
    return external_swaps(sentences, pos=pos, substitution_singular=substitution_singular, d0_s=d0_s, d1_s=d1_s,
                          thresh=thresh)
