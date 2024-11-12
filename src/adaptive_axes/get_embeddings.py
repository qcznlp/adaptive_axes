from minicons import cwe
import re
import torch
from sentence_transformers import SentenceTransformer


def extract_keyword_positions(sentence, keyword):
    """
    This function returns the start and end positions of a keyword in a sentence.
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = [(match.start(), match.end()) for match in pattern.finditer(sentence)]
    result = [(sentence, (start, end)) for start, end in matches]

    return result


def get_masked_sentence(sentence, keyword):
    """
    This function masks all keywords' occurrences in a sentence.
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    masked_sentence = pattern.sub('[MASK]', sentence)
    return masked_sentence


def get_BERT_embedding(model, sentence, keyword):
    """
    This function returns the BERT embedding of a keyword in a sentence.
    """
    keyword_positions = extract_keyword_positions(sentence, keyword)
    last_four = list(range(model.layers + 1))[-4:]
    representations = model.extract_representation(keyword_positions, layer=last_four)
    averaged_representations = []
    for representation in representations:
        averaged_representation = torch.mean(representation, dim=0)
        averaged_representations.append(averaged_representation)
    concatenated_representation = torch.cat(averaged_representations, dim=0)
    return concatenated_representation.numpy()


def get_context_text_embedding(model, sentence, keyword):
    masked_sentence = get_masked_sentence(sentence, keyword)
    vec = model.encode([masked_sentence])
    return vec[0]
