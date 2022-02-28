from relationextraction import KnowledgeTriplets

import pandas as pd

from pathlib import Path

import ndjson
import emoji
import spacy


def extract_triplets(text):
    """
    Extract triplets from a given text.
    """

    if not isinstance(text, list):
        text = [text]
    text = [t.strip() for t in text]
    triplets = rel_extractor.extract_relations(text)
    return triplets


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
                            "triplet": trip["extraction_3"][i],
                            "confidence": trip["confidence"][i],
                        }
                    )
