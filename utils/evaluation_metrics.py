"""
Created 31 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different evaluation metric functions
"""

import nltk
from utils.llm_functions import *
from utils.graph_functions import *
from utils.search_funcs import get_prediction


def edit_distance(a, b):
    """
    A function that computes the word-level levenshtein distance between two strings and returns it normalized
    by the length of the first string.

    :param a: string 1
    :param b: string 2

    :return: float value representing the normalized levenshtein distance between the two strings
    """
    a_lst = a.split()
    b_lst = b.split()
    lev = nltk.edit_distance(a_lst, b_lst)

    return lev / len(a_lst)


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
                pair_fluency = abs(
                    1 - sent_scoring(model, tokenizer, x[1], cuda=cuda)[0] / sent_scoring(
                        model, tokenizer, x[0], cuda=cuda)[0]
                )
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
