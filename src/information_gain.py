import math
from src.inference import inference_by_enumeration, marginal_probabilities

def entropy(distribution):
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)

def expected_information_gain(joint_dist, query_vector):
    """
    Computes the expected information gain for each unknown symptom.

    Parameters:
        joint_dist: the full joint probability distribution
        query_vector (list of int): One value for each variable.
            -2 = query variable
            -1 = hidden variable
            other = evidence value (0/1 for symptoms)

    Returns:
        dict: Mapping symptom index to information gain
    """
    current_dist = inference_by_enumeration(joint_dist, query_vector)
    current_entropy = entropy(current_dist)

    info_gains = {}
    marginals = marginal_probabilities(joint_dist)

    for i in range(4):  # Only symptoms s1-s4
        if query_vector[i] != -1:
            continue  # Skip known symptoms

        expected_entropy = 0.0
        for val in [0, 1]:
            temp_query = query_vector.copy()
            temp_query[i] = val
            dist = inference_by_enumeration(joint_dist, temp_query)
            p_val = marginals[i][val]
            expected_entropy += p_val * entropy(dist)

        info_gains[i] = current_entropy - expected_entropy

    return info_gains
