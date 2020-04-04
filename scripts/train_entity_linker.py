#!/usr/bin/env python
# coding: utf8

"""Example of training spaCy's entity linker, starting off with an
existing model and a pre-defined knowledge base.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* Entity Linking: https://spacy.io/usage/linguistic-features#entity-linking

Compatible with: spaCy v2.2.3
Last tested with: v2.2.3
"""
from __future__ import unicode_literals, print_function

import plac
import random
import os
import logging
from pathlib import Path
from tqdm import tqdm

from spacy.symbols import PERSON
from spacy.vocab import Vocab
from spacy.util import registry

from scispacy.data_util import med_mentions_example_iterator, read_full_med_mentions
from scispacy.umls_utils import UmlsKnowledgeBase

import spacy
from spacy.kb import KnowledgeBase
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse

from bin.wiki_entity_linking.entity_linker_evaluation import measure_performance

from scispacy.umls_linking import UmlsEntityLinker
from scispacy.data_util import MedMentionEntity

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# TODO: cleanup and style

def _get_gold_parse(doc, entities, dev, kb):
    gold_entities = {}
    tagged_ent_positions = {(ent.start_char, ent.end_char): ent for ent in doc.ents}

    for entity in entities:
        entity_id = entity.umls_id
        alias = entity.mention_text
        start = entity.start
        end = entity.end

        candidate_ids = []
        if kb and not dev:
            candidates = kb.get_candidates(alias)
            candidate_ids = [cand.entity_ for cand in candidates]

        # TODO: lots of data gets filtered out here, anything to be done?
        tagged_ent = tagged_ent_positions.get((start, end), None)
        if tagged_ent:
            should_add_ent = dev or entity_id in candidate_ids

            if should_add_ent:
                value_by_id = {entity_id: 1.0}
                if not dev:
                    random.shuffle(candidate_ids)
                    value_by_id.update(
                        {kb_id: 0.0 for kb_id in candidate_ids if kb_id != entity_id}
                    )
                gold_entities[(start, end)] = value_by_id

    return GoldParse(doc, links=gold_entities)


def read_el_docs_golds(nlp, examples, dev, kb):
    """ This method provides training/dev examples that correspond to the entity annotations found by the nlp object.
     For training, it will include both positive and negative examples by using the candidate generator from the kb.
     For testing (kb=None), it will include all positive examples only."""

    texts = []
    entities_list = []

    for example in examples:
        clean_text = example.text
        entities = example.entities

        texts.append(clean_text)
        entities_list.append(entities)

    docs = nlp.pipe(texts, batch_size=10)

    for doc, entities in tqdm(zip(docs, entities_list), desc="Creating docs...", total=len(texts)):
        for sentence in doc.sents:
            start_char = sentence.start_char
            end_char = sentence.end_char
            char_range = range(start_char, end_char)
            sentence_doc = sentence.as_doc()

            filtered_entities = []
            for entity in entities:
                if len(set(char_range).intersection(set(range(entity.start, entity.end)))) > 0:
                    new_entity = MedMentionEntity(start=entity.start-start_char, end=entity.end-start_char, mention_text=entity.mention_text, mention_type=entity.mention_type, umls_id=entity.umls_id)
                    filtered_entities.append(new_entity)

            gold = _get_gold_parse(sentence_doc, filtered_entities, dev=dev, kb=kb)

            if gold and len(gold.links) > 0:
                yield sentence_doc, gold


@plac.annotations(
    kb_path=("Path to the knowledge base", "positional", None, Path),
    vocab_path=("Path to vocab", "option", "v", Path),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(kb_path, vocab_path, output_dir=None, n_iter=50):
    """Create a blank model with the specified vocab, set up the pipeline and train the entity linker.
    The `vocab` should be the one used during creation of the KB."""
    #TODO: need to unify the vocab stuff
    vocab = Vocab().from_disk(vocab_path)
    # create blank Language class with correct vocab
    nlp = spacy.load("en_core_sci_md")
    nlp.vocab = vocab
    print("Loaded model")

    #TODO: try different params here
    linker = UmlsEntityLinker(
        resolve_abbreviations=False,
        k=100,
        threshold=0.5,
        no_definition_threshold=1.1,
        filter_for_definitions=True,
        max_entities_per_mention=40,
    )

    @registry.kb.register("get_candidates")
    def umls_get_candidates(kb, ent):
        doc = nlp.tokenizer(ent.text)
        doc.ents = [Span(doc, 0, len(doc), label="Entity")]
        linked_doc = linker(doc)
        all_candidates_from_candidate_generator = list(linked_doc.ents)[0]._.umls_ents
        all_candidates = []
        for candidate in all_candidates_from_candidate_generator:
            spacy_candidates = kb.get_candidates(
                linker.umls.cui_to_entity[candidate[0]].canonical_name
            )
            all_candidates += spacy_candidates

        return all_candidates

    # Create the Entity Linker component and add it to the pipeline.
    if "entity_linker" not in nlp.pipe_names:
        # use only the predicted EL score and not the prior probability (for demo purposes)
        #TODO: other el params?
        cfg = {"incl_prior": False}
        entity_linker = nlp.create_pipe("entity_linker", cfg)
        kb = KnowledgeBase(vocab=nlp.vocab)
        kb.load_bulk(kb_path)
        print("Loaded Knowledge Base from '%s'" % kb_path)
        entity_linker.set_kb(kb)
        nlp.add_pipe(entity_linker, last=True)

    corpus = os.path.join("/home/daniel/scispacy/med_mentions/")
    train, dev, test = read_full_med_mentions(corpus, spacy_format=False)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
    with nlp.disable_pipes(*other_pipes):  # only train Entity Linking
        optimizer = nlp.begin_training()
        #TODO: params?
        optimizer.learn_rate = 0.005
        optimizer.L2 = 1e-6

    print("Loading docs...")
    train_docs = list(read_el_docs_golds(nlp, train, False, kb))
    dev_docs = list(read_el_docs_golds(nlp, dev, True, kb))
    test_docs = list(read_el_docs_golds(nlp, test, True, kb))

    # get names of other pipes to disable them during training
    print("Starting training...")
    for itn in range(n_iter):
        random.shuffle(train_docs)
        losses = {}
        #TODO: params
        batches = minibatch(train_docs, size=compounding(8.0, 128.0, 1.001))
        batchnr = 0
        articles_processed = 0

        # we either process the whole training file, or just a part each epoch
        bar_total = len(train_docs)

        with tqdm(total=bar_total, leave=False, desc="Epoch " + str(itn)) as pbar:
            for batch in batches:
                with nlp.disable_pipes("entity_linker"):
                    docs, golds = zip(*batch)

                with nlp.disable_pipes(*other_pipes):
                    #TODO: params
                    nlp.update(
                        docs=docs,
                        golds=golds,
                        sgd=optimizer,
                        drop=0.5,
                        losses=losses,
                    )
                    batchnr += 1
                    articles_processed += len(docs)
                    pbar.update(len(docs))

        if batchnr > 0:
            logging.info(
                "Epoch {} trained on {} articles, train loss {}".format(
                    itn, articles_processed, round(losses["entity_linker"] / batchnr, 2)
                )
            )
            measure_performance(
                dev_docs,
                kb,
                entity_linker,
                baseline=False,
                context=True,
                dev_limit=None,
            )

    measure_performance(
        dev_docs,
        kb,
        entity_linker,
        baseline=False,
        context=True,
        dev_limit=None,
    )

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print()
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        measure_performance(
            dev_docs,
            kb,
            nlp2.get_pipe('entity_linker'),
            baseline=False,
            context=True,
            dev_limit=None,
        )


if __name__ == "__main__":
    plac.call(main)
