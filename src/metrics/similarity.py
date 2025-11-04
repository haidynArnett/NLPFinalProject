# src/metrics/similarity.py
# Similarity computation utilities for embeddings
# Cosine similarity, semantic similarity, and distance metrics
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_cosine_similarity_matrix(restatements, embed):
    restatement_embeddings = np.adarray([embed(restatement) for restatement in restatements])
    similarity_matrix = cosine_similarity(restatement_embeddings[np.newaxis,:,:], restatement_embeddings[:,np.newaxis,:])
    return similarity_matrix