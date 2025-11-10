# src/metrics/similarity.py
# Similarity computation utilities for embeddings
# Cosine similarity, semantic similarity, and distance metrics
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_cosine_similarity_matrix(restatements, embed):
    restatement_embeddings = np.array([embed(restatement) for restatement in restatements])
    similarity_matrix = np.array([[cosine_similarity(np.expand_dims(restatement_embeddings[i,:], 0), np.expand_dims(restatement_embeddings[j,:], 0)) for i in range(len(restatements))]for j in range(len(restatements))]).squeeze()
    return similarity_matrix

# calculates the cosine similarity between each hop for telephone test
def step_similarity(restatments, embed):
    sims = []
    for i in range(len(restatments) - 1):
        S = compute_cosine_similarity_matrix([restatments[i], restatments[i + 1]], embed)
        sims.append(S[0, 1])
    return sims

def levenshtein(a, b):
    # implement levenshtein
    return 

# distance between each hop for telephone test
def step_distance(restatements):
    dists = []
    for i in range(len(restatements) - 1):
        dists.append(levenshtein(restatements[i], restatements[i + 1]))
    return dists

def final_distance(restatements):
    return levenshtein(restatements[0], restatements[1])
