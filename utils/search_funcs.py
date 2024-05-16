from utils.gpt2_functions import *


def get_prediction(model, tokenizer, text, return_logits=False):
    """
    A function that takes as input a model, a tokenizer and a text and returns the prediction of the model.

    :param model: a pretrained model used for prediction
    :param tokenizer: the tokenizer used for encoding the text
    :param text: string used as input for the model
    :param return_logits: whether to return the logits or the prediction

    :return: the prediction of the model
    """
    # get encoded input tokens from text
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_text)

    return output.logits if return_logits else torch.argmax(output.logits).item()


def beam_search(text, substitutions, original_probs, original_fluency, model=None, tokenizer=None,
                fluency_model=None, fluency_tokenizer=None, beam_size=5, max_subs=10, use_contrastive_prob=False):
    """
    A function that uses beam search to create appropriate adversarials based on a given text.
    At each step, we take the top b candidates based on fluency, and we continue until the original
    prediction is changed.

    :param text: string representing the original sentence
    :param substitutions: dictionary with the all possible substitutions
    :param original_probs: the predicted logits of the original sentence
    :param original_fluency: the fluency of the original sentence
    :param model: the model used for prediction
    :param tokenizer: the tokenizer used for prediction
    :param fluency_model: the model used for fluency scoring
    :param fluency_tokenizer: the tokenizer used for fluency scoring
    :param beam_size: the size of the beam
    :param max_subs: the maximum number of substitutions allowed
    :param use_contrastive_prob: boolean value, for when to use contrastive probability as beam search criterion
    
    :return: string representing the adversarial sentence
    """

    # initialize model and tokenizer used for evaluating fluency if not provided
    if fluency_model is None or fluency_tokenizer is None:
        fluency_model, fluency_tokenizer = model_init('t5-base', cuda=not torch.cuda.is_available())

    # get the original prediction
    original_pred = torch.argmax(original_probs).item()
    sent = text.lower()
    cand_set = {(sent, 0, 0)}
    counter = 0
    new_candidates = []

    while counter < max_subs and cand_set:

        elem = cand_set.pop()
        candidate, prev_sub_idx, _ = elem
        candidate = candidate.split()

        for idx, word in enumerate(candidate[prev_sub_idx:]):
            new_cand = candidate.copy()
            new_cand[prev_sub_idx+idx] = substitutions.get(word, word)
            new_cand_str = " ".join(new_cand)

            # check if the new candidate is different from the original sentence
            if new_cand_str != " ".join(candidate):
                contrastive_prob = 0

                # get the prediction of the new candidate if a model is provided
                if model is not None and tokenizer is not None:
                    logits = get_prediction(model, tokenizer, new_cand_str, return_logits=True)
                    probs = torch.softmax(logits, dim=1)[0]
                    new_pred = torch.argmax(logits).item()

                    # if prediction is flipped, return the new candidate
                    if new_pred != original_pred:
                        return new_cand_str

                    # compute contrastive probability
                    if use_contrastive_prob:
                        contrastive_prob = original_probs[original_pred] - probs[original_pred]

                # else, add it to the candidate list
                new_candidates.append((new_cand_str, prev_sub_idx+idx, contrastive_prob))

        # sort the new candidates based on fluency and add the top b to the next round
        if new_candidates:
            new_candidates = sorted(
                new_candidates,
                key=lambda x: sent_scoring(fluency_model, fluency_tokenizer, x[0])[0] / original_fluency + x[2],
                reverse=True
            )
            cand_set = cand_set.union(new_candidates[:min(beam_size, len(new_candidates))])
        else:
            break

        counter += 1  # update counter value
        new_candidates = []  # reset the new candidates list

    # if no adversarial is found, return the best candidate based on fluency
    try:
        best_cand = max(cand_set,
                        key=lambda x: sent_scoring(fluency_model, fluency_tokenizer, x[0])[0] / original_fluency + x[2]
                        )[0]
    except ValueError:
        best_cand = sent

    return best_cand
