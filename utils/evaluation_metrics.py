"""
Created 31 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different functions to evaluate the generated counterfactuals
"""

import numpy as np
import pandas as pd

# Metric-related imports
from joblib import Parallel, delayed
from pylev import levenshtein as lev_dist

from utils.gpt2_functions import *
from utils.graph_functions import *


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
        if len(x[0]) <= 1024 and len(x[1]) <= 1024:
            try:
                pair_fluency = abs(sent_scoring(model, tokenizer, x[0], cuda=cuda)[0] -
                                   sent_scoring(model, tokenizer, x[1], cuda=cuda)[0])
                avg_fluency += pair_fluency if pair_fluency is not None else 0
                counter += 1
            except RuntimeError:
                continue

    # avg_fluency = sum(
    #     Parallel(n_jobs=-3)(
    #         delayed(
    #             lambda x: abs(sent_scoring(gpt_model, gpt_tokenizer, x[0], cuda=cuda)[0] -
    #             sent_scoring(gpt_model, gpt_tokenizer, x[1], cuda=cuda)[0]))(x) for x in sent_pairs
    #     if len(x[0]) <= 1024 and len(x[1]) <= 1024
    #     )
    # )

    # counter = sum(
    #     Parallel(n_jobs=-3)(
    #         delayed(
    #             lambda x: (len(x[0]) <= 1024 and len(x[1]) <= 1024))(x) for x in sent_pairs
    #     )
    # )

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
    sentences = [elem[0] for elem in data.values.tolist()]
    counter_sentences = [elem[0] for elem in counter_data.values.tolist()]

    assert len(sentences) == len(counter_sentences)

    # compute average levenshtein distance as a measurement of closeness
    avg_lev = sum(Parallel(n_jobs=-1)(delayed(lev_dist)(x[0], x[1]) for x in zip(sentences, counter_sentences))) / len(
        sentences)

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
    sentences = [elem[0] for elem in data.values.tolist()]
    counter_sentences = [elem[0] for elem in counter_data.values.tolist()]

    assert len(sentences) == len(counter_sentences)

    # compute average bertscore
    avg_bertscore = sum(
        bertscore.compute(predictions=counter_sentences, references=sentences, model_type="distilbert-base-uncased")[
            'f1']) / len(sentences)

    return avg_bertscore


def get_flip_rate(original_p, counter_p):
    """
    A function that takes as input the original predictions and the new ones, and returns the
    flip-rate as a percentage.

    :param original_p: list containing the predictions for the original data
    :param counter_p: dataframe containing the predictions for the counter data
    :returns: dictionary containing model-related metrics
    """

    # check that predictions and counter_predictions are of the same length
    assert len(original_p) == len(counter_p)

    # compute flip_rate
    flip_rate_percent = sum(
        Parallel(n_jobs=-1)(delayed(lambda x: x[0] != x[1])(x) for x in zip(original_p, counter_p))) / len(original_p)

    return flip_rate_percent


def get_baseline_metric(data, pos, eval_metric, model_required=False, preprocessor=None, model=None, antonyms=False):
    """
    A function that takes as input a dataframe with the textual data, and computes a metric based on a bipartite graph,
    where the edge weights represent the distance between words (nodes) as extracted from wordnet.

    :param data: pd.DataFrame() containing one column with the textual data
    :param pos: string that specifies which part-of-speech shall be considered for substitution (noun, verb, adv)
    :param eval_metric: a function that computes the metric which must be optimized during fine-tuning
    :param model_required: boolean value specifing whether a pretrained model is also required for the metric
    computation
    :param preprocessor: a custom class that implements the necessary preprocessing of the data
    :param model: a pretrained model on the dataset
    :param antonyms: boolean value specifying whether antonyms should be considered for substitution
    :returns: a float value representing the computed evaluation metric
    """

    sents = [elem[0] for elem in data.values.tolist()]
    counter_sents, _, _, _ = get_edits(sents, pos=pos, thresh=3, antonyms=antonyms)

    counter_data_df = pd.DataFrame({
        'counter_sents': counter_sents
    })

    # print('Generating Model Agnostic Metrics...')
    # metrics = generate_model_agnostic_metrics(data, counter_data_df)

    if not model_required:
        return eval_metric(data, counter_data_df), counter_data_df

    else:
        # first process the original data and get model predictions
        processed_data = preprocessor.process(data)
        original_preds = model.predict(processed_data)

        # do the same but for the counterfactual-generated data
        processed_counter_data = preprocessor.process(counter_data_df)
        counter_preds = model.predict(processed_counter_data)

        return eval_metric(original_preds, counter_preds), counter_data_df


