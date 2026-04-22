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
]

model = SentenceTransformer(args.model)

encoded = model.encode(sentences, normalize_embeddings=True)

print(f"""
        vec![
            vec!{[float(round(x*1_000_000)/1_000_000) for x in encoded[0][:5]]},
            vec!{[float(round(x*1_000_000)/1_000_000) for x in encoded[1][:5]]}
        ]
""")