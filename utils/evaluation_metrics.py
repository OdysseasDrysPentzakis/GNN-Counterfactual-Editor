"""
Created 31 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different functions to evaluate the generated counterfactuals
"""

import numpy as np
import pandas as pd

# Metric-related imports
import nltk
from evaluate import load

from utils.gpt2_functions import *
from utils.graph_functions import *
from utils.search_funcs import get_prediction


def edit_distance(a, b):
    a_lst = a.split()
    b_lst = b.split()
    lev = nltk.edit_distance(a_lst, b_lst)

    return lev / len(a_lst)


def adversarial_success(original_output, counterfactual_output):
    """
    :param original_output: a list with the outputs produced from the original sentences
    :param counterfactual_output: a list with the outputs produced from the generated counterfactuals
    :return: a float representing the per-word-influence of each intervention on the outcome change
    """

    # check that the two list are of equal length
    assert len(original_output) == len(counterfactual_output)

    return np.mean([((t[0] - t[1]) / t[0]) for t in zip(original_output, counterfactual_output)]) / len(original_output)


def get_fluency(data, counter_data, model, tokenizer):
    """
    A function that takes as input the original and the counter data and returns the average fluency
    between the sentence pairs

    :param data: dataframe containing one column with the original data
    :param counter_data: dataframe containing one column with the counter data
    :param model: LL model for fluency scoring
    :param tokenizer: pretrained tokenizer for splitting sentences into tokens
    :returns: float value representing the average fluency
    """

    # extract sentences and counter-sentences from the data and check that they are of the same length
    sentences = [elem[0] for elem in data.values.tolist()]
    counter_sentences = [elem[0] for elem in counter_data.values.tolist()]

    assert len(sentences) == len(counter_sentences)

    # compute average fluency
    cuda = not torch.cuda.is_available()
    sent_pairs = zip(sentences, counter_sentences)

    avg_fluency, counter = 0, 0
    for x in tqdm(sent_pairs):
        if type(x[1]) != float and len(x[0]) <= 1024 and len(x[1]) <= 1024:
            try:
                pair_fluency = sent_scoring(model, tokenizer, x[1], cuda=cuda)[0] / sent_scoring(
                    model, tokenizer, x[0], cuda=cuda)[0]
                avg_fluency += pair_fluency if pair_fluency is not None else 0
                counter += 1
            except RuntimeError:
                continue

    return avg_fluency / counter


def get_closeness(data, counter_data):
    """
    A function that takes as input the original and the counter data and returns the average levenshtein
    distance as a measure of closeness between the sentence pairs

    :param data: dataframe containing one column with the original data
    :param counter_data: dataframe containing one column with the counter data
    :returns: float value representing the average levenshtein distance
    """

    # extract sentences and counter-sentences from the data and check that they are of the same length
    dirty_sentences = [elem[0] for elem in data.values.tolist()]
    dirty_counter_sentences = [elem[0] for elem in counter_data.values.tolist()]

    # filter out sentences where the counter sentence is NaN
    sentences, counter_sentences = [], []
    for idx in range(len(dirty_sentences)):
        if type(dirty_counter_sentences[idx]) != float:
            sentences.append(dirty_sentences[idx].lower())
            counter_sentences.append(dirty_counter_sentences[idx])

    assert len(sentences) == len(counter_sentences)

    # compute average levenshtein distance as a measurement of closeness
    avg_lev = sum(edit_distance(x[0].lower(), x[1]) for x in zip(sentences, counter_sentences) if
                  type(x[1]) != float) / len(sentences)

    return avg_lev


def get_bertscore(data, counter_data, bertscore):
    """
    A function that takes as input the original and the counter data and returns the average bertscore
    between the sentence pairs

    :param data: dataframe containing one column with the original data
    :param counter_data: dataframe containing one column with the counter data
    :param bertscore: bertscore object for computing the bertscore similarity
    :returns: float value representing the average bertscore
    """

    # extract sentences and counter-sentences from the data and check that they are of the same length
    dirty_sentences = [elem[0] for elem in data.values.tolist()]
    dirty_counter_sentences = [elem[0] for elem in counter_data.values.tolist()]

    # filter out sentences where the counter sentence is NaN
    sentences, counter_sentences = [], []
    for idx in range(len(dirty_sentences)):
        if type(dirty_counter_sentences[idx]) != float:
            sentences.append(dirty_sentences[idx])
            counter_sentences.append(dirty_counter_sentences[idx])

    assert len(sentences) == len(counter_sentences)

    # compute average bertscore
    avg_bertscore = sum(
        bertscore.compute(predictions=counter_sentences, references=sentences, model_type="distilbert-base-uncased")[
            'f1']) / len(sentences)

    return avg_bertscore


def get_flip_rate(data, counter_data, model, tokenizer):
    """
    A function that takes as input the original sentences and the counter sentences, and returns the
    flip-rate as a percentage.

    :param data: dataframe containing one column with the original data
    :param counter_data: dataframe containing one column with the counter data
    :param model: a pretrained model on the dataset
    :param tokenizer: a pretrained tokenizer for the model

    :returns: float value representing the flip-rate metric
    """

    # extract sentences and counter-sentences from the data and check that they are of the same length
    dirty_sentences = [elem[0] for elem in data.values.tolist()]
    dirty_counter_sentences = [elem[0] for elem in counter_data.values.tolist()]

    # filter out sentences where the counter sentence is NaN
    sentences, counter_sentences = [], []
    for idx in range(len(dirty_sentences)):
        if type(dirty_counter_sentences[idx]) != float:
            sentences.append(dirty_sentences[idx])
            counter_sentences.append(dirty_counter_sentences[idx])

    assert len(sentences) == len(counter_sentences)

    flip_rate = 0
    for idx in range(len(sentences)):
        if get_prediction(model, tokenizer, sentences[idx]) != get_prediction(model, tokenizer, counter_sentences[idx]):
            flip_rate += 1

    return flip_rate / len(sentences)


def get_baseline_metric(data, pos, eval_metric, model_required=False, preprocessor=None, model=None, antonyms=False):
    """
    A function that takes as input a dataframe with the textual data, and computes a metric based on a bipartite graph,
    where the edge weights represent the distance between words (nodes) as extracted from wordnet.

    :param data: pd.DataFrame() containing one column with the textual data
    :param pos: string that specifies which part-of-speech shall be considered for substitution (noun, verb, adj)
    :param eval_metric: a string that represents the metric which must be optimized during fine-tuning
    :param model_required: boolean value specifing whether a pretrained model is also required for the metric
    computation
    :param preprocessor: a custom class that implements the necessary preprocessing of the data
    :param model: a pretrained model on the dataset
    :param antonyms: boolean value specifying whether antonyms should be considered for substitution
    :returns: a float value representing the computed evaluation metric
    """

    # create counter sentences
    sents = [elem[0] for elem in data.values.tolist()]
    counter_sents, _, _, _ = get_edits(sents, pos=pos, thresh=3, antonyms=antonyms)

    # convert them to a dataframe
    counter_data_df = pd.DataFrame({
        'counter_sents': counter_sents
    })

    # initialize the required parameters of the evaluation metric function
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

    # compute the specified metric
    if not model_required:

        if eval_metric == 'fluency':
            return get_fluency(data, counter_data_df, fluency_model, fluency_tokenizer), counter_data_df
        elif eval_metric == 'bertscore':
            return get_bertscore(data, counter_data_df, bertscore), counter_data_df
        elif eval_metric == 'fluency_bertscore':
            fluency = get_fluency(data, counter_data_df, fluency_model, fluency_tokenizer)
            bscore = get_bertscore(data, counter_data_df, bertscore)
            return 2 * fluency * bscore / (fluency + bscore), counter_data_df
        elif eval_metric == 'closeness':
            return get_closeness(data, counter_data_df), counter_data_df

    else:
        # first process the original data and get model predictions
        processed_data = preprocessor.process(data)
        original_preds = model.predict(processed_data)

        # do the same but for the counterfactual-generated data
        processed_counter_data = preprocessor.process(counter_data_df)
        counter_preds = model.predict(processed_counter_data)

        return eval_metric(original_preds, counter_preds), counter_data_df
