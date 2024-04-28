import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5ForConditionalGeneration, T5Tokenizer


def model_init(model_string='gpt2', cuda=False):
    """
    A function that initializes a LM and a Tokenizer based on GPT2.

    :param model_string: string representing the base model for the transformer and the tokenizer
    :param cuda: boolean value, determining whether to use gpu for model inference
    :return: the pretrained model and tokenizer
    """

    if "gpt2" in model_string:
        tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        model = GPT2LMHeadModel.from_pretrained(model_string)
    elif "t5-base" in model_string:
        tokenizer = T5Tokenizer.from_pretrained(model_string)
        model = T5ForConditionalGeneration.from_pretrained(model_string)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)

    model.eval()

    if cuda:
        try:
            model.to('cuda')
            print("Model to gpu")
        except ValueError:
            pass
        except RuntimeError:
            pass

    return model, tokenizer


def sent_scoring(model, tokenizer, text, cuda=False):
    """
    A function that uses the given LLM and Tokenizer to compute the probability of a given sentence.

    :param model: a pretrained transformer model
    :param tokenizer: a pretrained tokenizer
    :param text: a string representing the sentence whose probability will be computed
    :param cuda: boolean value, determining whether to use gpu for model inference
    :return: the computed loss of the sentence and log_probability of the last token
    """

    assert model is not None
    assert tokenizer is not None

    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

    if cuda:
        tokens.to('cuda')

    with torch.no_grad():
        outputs = model(tokens, labels=tokens)

    loss, logits = outputs[:2]
    loss, log_prob = loss.item(), logits[0, -1, tokens[0, -1]].item()

    # clear memory
    del tokens
    del logits
    del outputs

    if cuda:
        torch.cuda.empty_cache()

    return loss, log_prob
