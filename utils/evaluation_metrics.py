"""
Created 31 May 2023
@author: Dimitris Lymperopoulos
Description: A python file containing different functions to evaluate the generated counterfactuals
"""

import numpy as np


def adversarial_success(original_output, counterfactual_output):
    """
    :param original_output: a list with the outputs produced from the original sentences
    :param counterfactual_output: a list with the outputs produced from the generated counterfactuals
    :return: a float representing the per-word-influence of each intervention on the outcome change
    """

    # check that the two list are of equal length
    assert len(original_output) == len(counterfactual_output)

    return np.mean([((t[0] - t[1]) / t[0]) for t in zip(original_output, counterfactual_output)]) / len(original_output)


def main():

    # check if adversarial success returns the correct value
    assert adversarial_success([1, 1, 1, 1], [0, 0, 0, 0]) == 0.25   # average outcome change is 1 --> 1/4 = 0.25
    print("Adversarial Success works correctly!")


if __name__ == "__main__":
    main()
