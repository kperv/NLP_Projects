import spacy
import numpy as np
import json
from collections import defaultdict
from collections import namedtuple


# vectors with cosine similarity below are considered different
lower_threshold = 0.1

# vectors with cosine similarity above this value are considered similar
# and of the same group
upper_threshold = 0.4

# used to compare mean similarities from different groups to see
# if they are too close to take max of them in this case the noun-vector is
# identically close to any group and considered not similar
variance_threshold = 0.15


ReprVectors = namedtuple("ReprVectors", "word_groups group_names")


def to_json(entities, filepath):
    entities_from_doc = {key: list(value) for key, value in entities.items()}
    result = {"entities": entities_from_doc}
    with open(filepath, 'w') as output_file:
        json.dump(result, output_file, indent=2)


def representative_vectors(nlp):
    # create labelled words to use later for cosine similarity
    return ReprVectors(
        word_groups=[nlp('mother dog hand tree actress snake'),
                     nlp('country shop park room home school'),
                     nlp('ball food money bag dress car')],
        group_names='living_entity place object'.split())


def get_input_doc(filepath):
    with open(filepath, 'r') as input_file:
        doc = input_file.read()
    return doc


def make_tokens_and_check_ner(doc, doc_entities, group_names):
    tokens = set()
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                doc_entities['living_entity'].add(ent.text)

        # check for names again and collect all nouns into a set
        for token in sent:
            if token.pos_ == "PROPN":
                doc_entities['living_entity'].add(token.text)
            elif token.pos_ == "NOUN":
                tokens.add(token.lemma_)
    return tokens


def find_similarities(nlp, tokens, doc_entities, repr_vecs):
    word_groups, group_names = repr_vecs
    # a 2d array to be filled with values of word similarities
    # between a word vector for an unlabelled noun and all example entities
    similarity_matrix = np.zeros((len(word_groups),
                                  len(word_groups[0])))
    for token in tokens:
        for i, group in enumerate(word_groups):
            for j, example_token in enumerate(group):
                similarity_matrix[i, j] = nlp(token).similarity(example_token)

        max_similarity = np.amax(similarity_matrix)
        group_index, _ = np.where(similarity_matrix == max_similarity)
        group_index = group_index.item()

        if max_similarity < lower_threshold:
            continue
        elif max_similarity >= upper_threshold:
            doc_entities[group_names[group_index]].add(token)
        else:
            mean_group_similarity = np.mean(similarity_matrix, axis=1)
            if np.var(mean_group_similarity) >= variance_threshold:
                group_index = np.argmax(mean_group_similarity)
                doc_entities[group_names[group_index]].add(token)


def main():
    nlp = spacy.load("en_core_web_lg")
    repr_vecs = representative_vectors(nlp)

    doc = get_input_doc('input.txt')
    doc = doc.lower()
    doc = nlp(doc)

    doc_entities = defaultdict(set)
    tokens = make_tokens_and_check_ner(doc, doc_entities, repr_vecs.group_names)
    find_similarities(nlp, tokens, doc_entities, repr_vecs)
    to_json(doc_entities, 'output.txt')


if __name__ == "__main__":
    main()





