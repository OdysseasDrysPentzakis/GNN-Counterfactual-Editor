from utils.gpt2_functions import *


def beam_search(text, substitutions, original_pred, original_fluency, model, beam_size=3, max_subs=10):
    """
    A function that uses beam search to create appropriate adversarials based on a given text.
    At each step, we take the top b candidates based on fluency and we continue until the original
    prediction is changed.

    :param text: string representing the original sentence
    :param substitutions: dictionary with the all possible substitutions
    :param original_pred: the prediction of the original sentence
    :param original_fluency: the fluency of the original sentence
    :param model: the model used for prediction
    :param beam_size: the size of the beam
    :param max_subs: the maximum number of substitutions allowed
    
    :return: string representing the adversarial sentence
    """

    # initialize model and tokenizer used for evaluating fluency
    fluency_model, fluency_tokenizer = model_init('gpt2', cuda=not torch.cuda.is_available())

    sent = text.lower()
    cand_set = {(sent, 0)}
    counter = 0
    new_candidates = []

    while counter < max_subs and cand_set:

        elem = cand_set.pop()
        candidate, prev_sub_idx = elem
        candidate = candidate.split()

        for idx, word in enumerate(candidate[prev_sub_idx:]):
            new_cand = candidate.copy()
            new_cand[prev_sub_idx+idx] = substitutions.get(word, word)

            # check if the new candidate is different from the original sentence
            if " ".join(new_cand) != " ".join(candidate):
                # get the prediction of the new candidate if a model is provided
                if model is not None:
                    new_pred = model(new_cand)
                    # if prediction is flipped, return the new candidate
                    if new_pred != original_pred:
                        return " ".join(new_cand)

                # else, add it to the candidate list
                new_candidates.append((" ".join(new_cand), prev_sub_idx+idx))

        # sort the new candidates based on fluency and add the top b to the next round
        if new_candidates:
            new_candidates = sorted(
                new_candidates, key=lambda x: abs(original_fluency - sent_scoring(fluency_model, fluency_tokenizer, x[0])[0]),
                reverse=True
            )
            cand_set = cand_set.union(new_candidates[:min(beam_size, len(new_candidates))])
        else:
            break

        counter += 1  # update counter value
        new_candidates = []  # reset the new candidates list

    # if no adversarial is found, return the best candidate based on fluency
    best_candidate = max(cand_set,
                         key=lambda x: abs(original_fluency - sent_scoring(fluency_model, fluency_tokenizer, x[0])[0])
                         )[0]

    return best_candidate
