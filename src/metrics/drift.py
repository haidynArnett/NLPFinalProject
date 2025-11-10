# src/metrics/drift.py
# Metrics for measuring information drift in agent handoffs
# Includes drift velocity, acceleration, and similarity tracking
from .similarity import compute_cosine_similarity_matrix

# gets similarity to the original at each hop
def sim_to_original(restatements,embed):
    S = compute_cosine_similarity_matrix(restatements, embed)
    similarities= []
    for j in range(1, S.shape[0]):
        value = float(S[0, j])
        similarities.append(value)
    return similarities

# similarity between each hop
def step_similarity(restatements,embed):
    n = len(restatements)
    sims = []
    for i in range(n - 1):
        pair = [restatements[i], restatements[i + 1]]
        S_pair = compute_cosine_similarity_matrix(pair, embed)
        value = float(S_pair[0, 1])
        sims.append(value)
    return sims


def drift_velocity(similarities):
    velocities = []
    for i in range(1, len(similarities)):
        v = similarities[i] - similarities[i - 1]
        velocities.append(float(v))
    return velocities


def drift_acceleration(velocities):
    accelerations = []
    for i in range(1, len(velocities)):
        a = velocities[i] - velocities[i - 1]
        accelerations.append(float(a))
    return accelerations