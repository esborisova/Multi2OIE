from relationextraction import SpacyRelationExtractor
import spacy


if __name__ == "__main__":
    nlp = spacy.load("da_core_news_sm")

    test_sents = [
        "Pernille Blume vinder delt EM-sølv i Ungarn.",
        "Pernille Blume blev nummer to ved EM på langbane i disciplinen 50 meter fri.",
        "Hurtigst var til gengæld hollænderen Ranomi Kromowidjojo, der sikrede sig guldet i tiden 23,97 sekunder.",
        "Og at formen er til en EM-sølvmedalje tegner godt, siger Pernille Blume med tanke på, at hun få uger siden var smittet med corona.",
        "Ved EM tirsdag blev det ikke til medalje for den danske medley for mixede hold i 4 x 200 meter fri.",
        "In a phone call on Monday, Mr. Biden warned Mr. Netanyahu that he could fend off criticism of the Gaza strikes for only so long, according to two people familiar with the call",
        "That phone call and others since the fighting started last week reflect Mr. Biden and Mr. Netanyahu’s complicated 40-year relationship.",
        "Politiet skal etterforske Siv Jensen etter mulig smittevernsbrudd.",
        "En av Belgiens mest framträdande virusexperter har flyttats med sin familj till skyddat boende efter hot från en beväpnad högerextremist.",
    ]

    config = {"confidence_threshold": 1.0, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)

    pipe = nlp.pipe(test_sents)

    for d in pipe:
        print(d.text, "\n", d._.relation_triplets)
