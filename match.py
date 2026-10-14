#!/usr/bin/env python3
"""
Product match checker using EmbeddingGemma 2.
"""
import argparse
from sentence_transformers import SentenceTransformer
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference image path")
    parser.add_argument("--comp", required=True, help="Comparison image or video path")
    args = parser.parse_args()

    print("Loading EmbeddingGemma 2...")
    model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"audio_config": None})

    print("Embedding reference...")
    ref_emb = model.encode({"image": args.ref})

    print("Embedding comparison...")
    if args.comp.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        comp_emb = model.encode({"video": args.comp})
    else:
        comp_emb = model.encode({"image": args.comp})

    similarity = model.similarity(ref_emb, comp_emb).item()
    percentage = max(0, min(100, (similarity + 1) / 2 * 100))

    print(f"\nMatch Percentage: {percentage:.1f}%")
    print(f"Cosine similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()
