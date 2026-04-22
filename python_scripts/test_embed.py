from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer

parser = ArgumentParser(
    prog="yuzu_tests",
    description="Format outputs of sentence-transformers",
)

parser.add_argument(
    "model",
    choices=["Qwen/Qwen3-Embedding-0.6B", "NeuML/pubmedbert-base-embeddings-100K"],
)
args = parser.parse_args()

sentences = [
    "Yuzu is a citrus fruit and plant in the family Rutaceae of Chinese origin.",
    "Le yuzu est un agrume acide originaire de l'est de l'Asie.",
    "Yuzukoshō (柚子胡椒 / ゆずこしょう?, aussi yuzugoshō) est un type d'assaisonnement nippon.",
    "Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning.",
    "Le Cochon est un documentaire français de moyen métrage coréalisé par Jean-Michel Barjol et Jean Eustache, sorti en 1970.",
    "Le Cochon ('The Pig') is a fifty-minute featurette co-directed by Jean Eustache and Jean-Michel Barjol in 1970.",
]

model = SentenceTransformer(args.model)

encoded = model.encode(sentences, normalize_embeddings=True, batch_size=32)

print(
    "vec!["
    + ",\n".join(
        [
            "vec!"
            + str([float(round(x * 100_000) / 100_000) for x in encoded[row][:5]])
            for row in range(len(sentences))
        ]
    )
    + "]"
)
