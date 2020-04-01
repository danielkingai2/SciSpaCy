#!/usr/bin/env python
# coding: utf8

"""Example of defining and (pre)training spaCy's knowledge base,
which is needed to implement entity linking functionality.

For more details, see the documentation:
* Knowledge base: https://spacy.io/api/kb
* Entity Linking: https://spacy.io/usage/linguistic-features#entity-linking

Compatible with: spaCy v2.2.3
Last tested with: v2.2.3
"""
from __future__ import unicode_literals, print_function

import plac
import os
from pathlib import Path

from spacy.vocab import Vocab
import spacy
from spacy.kb import KnowledgeBase

from scispacy.data_util import med_mentions_example_iterator, read_full_med_mentions
from scispacy.umls_utils import UmlsKnowledgeBase

from bin.wiki_entity_linking.train_descriptions import EntityEncoder


INPUT_DIM = 300  # dimension of pretrained input vectors
DESC_WIDTH = 64  # dimension of output entity vectors


@plac.annotations(
    model=(
        "Model name, should have pretrained word embeddings",
        "positional",
        None,
        str,
    ),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=50):
    """Load the model, create the KB and pretrain the entity encodings.
    If an output_dir is provided, the KB will be stored there in a file 'kb'.
    The updated vocab will also be written to a directory in the output_dir."""

    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)

    # check the length of the nlp vectors
    if "vectors" not in nlp.meta or not nlp.vocab.vectors.size:
        raise ValueError(
            "The `nlp` object should have access to pretrained word vectors, "
            " cf. https://spacy.io/usage/models#languages."
        )

    kb = KnowledgeBase(vocab=nlp.vocab)

    corpus = os.path.join("/home/daniel/scispacy/med_mentions/")
    train, dev, test = read_full_med_mentions(corpus, spacy_format=False)

    cui_to_freq = {}
    for example in train:
        for entity in example.entities:
            cui = entity.umls_id
            if cui in cui_to_freq:
                cui_to_freq[cui] += 1
            else:
                cui_to_freq[cui] = 1

    umls_kb = UmlsKnowledgeBase()

    # set up the data
    entity_ids = []
    descriptions = []
    freqs = []
    for cui, umls_entity in umls_kb.cui_to_entity.items():
        freq = cui_to_freq.get(cui, 0)
        if umls_entity.definition is None:
            continue
        else:
            entity_ids.append(cui)
            freqs.append(freq)
            descriptions.append(umls_entity.definition)

    print("Train encoder...")
    # training entity description encodings
    # this part can easily be replaced with a custom entity encoder
    encoder = EntityEncoder(
        nlp=nlp, input_dim=INPUT_DIM, desc_width=DESC_WIDTH, epochs=n_iter,
    )
    encoder.train(description_list=descriptions, to_print=True)

    print("Applying encoder...")
    # get the pretrained entity vectors
    embeddings = encoder.apply_encoder(descriptions)

    print("Setting entities...")
    # set the entities, can also be done by calling `kb.add_entity` for each entity
    kb.set_entities(entity_list=entity_ids, freq_list=freqs, vector_list=embeddings)

    # TODO
    # adding aliases, the entities need to be defined in the KB beforehand
    cui_set = set(entity_ids)
    for alias, cuis in umls_kb.alias_to_cuis.items():
        filtered_cuis = [cui for cui in cuis if cui in cui_set]
        if filtered_cuis == []:
            continue
        kb.add_alias(
            alias=alias,
            entities=filtered_cuis,
            probabilities=[(1/len(filtered_cuis))]*len(filtered_cuis),  # the sum of these probabilities should not exceed 1
        )

    # test the trained model
    print()
    _print_kb(kb)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        kb_path = str(output_dir / "kb")
        kb.dump(kb_path)
        print()
        print("Saved KB to", kb_path)

        vocab_path = output_dir / "vocab"
        kb.vocab.to_disk(vocab_path)
        print("Saved vocab to", vocab_path)

        print()

        # test the saved model
        # always reload a knowledge base with the same vocab instance!
        print("Loading vocab from", vocab_path)
        print("Loading KB from", kb_path)
        vocab2 = Vocab().from_disk(vocab_path)
        kb2 = KnowledgeBase(vocab=vocab2)
        kb2.load_bulk(kb_path)
        _print_kb(kb2)
        print()


def _print_kb(kb):
    print(kb.get_size_entities(), "kb entities:", kb.get_entity_strings())
    print(kb.get_size_aliases(), "kb aliases:", kb.get_alias_strings())


if __name__ == "__main__":
    plac.call(main)

    # Expected output:

    # 2 kb entities: ['Q2146908', 'Q7381115']
    # 1 kb aliases: ['Russ Cochran']
