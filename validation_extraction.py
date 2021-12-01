from relationextraction import KnowledgeTriplets

import pandas as pd

from pathlib import Path

# import ndjson
# import emoji
import spacy

from spacy_transformers.align import get_alignment
from spacy_transformers.truncate import truncate_oversize_splits

from transformers import AutoTokenizer

rel_extractor = KnowledgeTriplets()

text = [
    "Lasse er en sød fyr på 27 år",
    "Han bor i Aarhus og har en kæreste der hedder Solvej som også har en hund der hedder Jens",
]
text = ["Hej mit navn er John og jeg bor i Svendborg"]
triplets = rel_extractor.extract_relations(text)
triplets["wordpieces"]


nlp = spacy.load("da_core_news_lg")
docs = nlp.pipe(text)

spacy_tokens = [doc for doc in docs]
wordpieces = triplets["wordpieces"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

align = get_alignment(spacy_tokens, wordpieces, tokenizer.all_special_tokens)


wp2tokid = []
for i, align_ in enumerate(zip(align.data, align.lengths)):
    tok_idx, l = align_
    wp2tokid += l * [i]


wp2tokid = {int(wp_id): t_id for wp_id, t_id in zip(align.data, wp2tokid)}

spacy_tokens[0][3 : 6 + 1]

#### TODO
###  Vi har nu mappping fra wp til spacy tokens (og omvendt)
###
### Per document:
###     Per triplet in triplet_span:
###         Per span in triplet:
###             convert fra starten af span til slut -> add til docs span
### def wpspan_to_token_span()

import torch
import numpy as np

wp = np.array(wordpieces)


from spacy_transformers.data_classes import FullTransformerBatch


tf_batch = FullTransformerBatch(
    spans=spacy_tokens, wordpieces=wordpieces, model_output=0, align=align
)


def doc_wp2tokid_getter(doc: Doc, bos=True, eos=True) -> List:
    """
    extract the wordpiece2tokenID mapping from a doc
    create a mapping from wordpieces to tokens id in the form of a list
    e.g.
    [0, 1, 1, 1, 2]
    indicate that there are three tokens (0, 1, 2) and token 1 consist of three
    wordpieces
    note: this only works under the assumption that the word pieces
    are trained using similar tokens. (e.g. split by whitespace)
    example:
    Doc.set_extension("wp2tokid", getter=doc_wp2tokid_getter)
    tokid = doc._.wp2tokid[2]
    token = doc[tokid]
    """
    wp2tokid = []
    tok = 0
    if bos is True:
        wp2tokid.append(None)
    for i in doc._.trf_data.align.lengths:
        wp2tokid += [tok] * i
        tok += 1
    if eos is True:
        wp2tokid.append(None)
    return wp2tokid


def extract_triplets(text):
    """
    Extract triplets from a given text.
    """

    if not isinstance(text, list):
        text = [text]
    text = [t.strip() for t in text]
    triplets = rel_extractor.extract_relations(text)
    return triplets


def flip_pos(t: tuple):
    if t:
        t[0], t[1] = t[1], t[0]
    return t


def remove_emojis(text):
    return emoji.replace_emoji(text, "")


if __name__ == "__main__":
    file_dir = Path().cwd() / "crowdtangle"
    files = file_dir.glob("*.ndjson")

    df = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(df)
    df = df.dropna(subset=["message"])
    df["message"] = df["message"].apply(remove_emojis)

    texts = df["message"].tolist()

    rel_extractor = KnowledgeTriplets()

    nlp = spacy.load("da_core_news_lg")

    with open("triplets.ndjson", "w") as f:
        writer = ndjson.writer(f, ensure_ascii=False)

        for doc in nlp.pipe(texts, disable=["ner"]):
            for sent in doc.sents:
                trip = extract_triplets(sent.text)
                for i in range(len(trip["sentence"])):
                    writer.writerow(
                        {
                            "sentence": trip["sentence"][i],
                            "triplet": flip_pos(trip["extraction_3"][i]),
                            "confidence": trip["confidence"][i],
                        }
                    )
