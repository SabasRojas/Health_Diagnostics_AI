import numpy as np

def inference_by_enumeration(joint_probs, query_vector):
    """
    Performs inference by enumeration.

    Parameters:
        joint_probs (numpy.ndarray): 5D array with joint probabilities.
        query_vector (list of int): One value for each variable.
            -2 = query variable
            -1 = hidden variable
            other = evidence value (0/1 for symptoms, 0-3 for condition)

    Returns:
        dict: A dictionary with possible values of the query variable and their probabilities.
    """

    # Step 1: Identify the query variable (should be only one -2)
    if query_vector.count(-2) != 1:
        raise ValueError("Exactly one query variable (-2) is required.")
    
    query_index = query_vector.index(-2)

    # Step 2: Loop through all possible values of the query variable
    result = {}
    value_range = range(4) if query_index == 4 else range(2)

    for q_val in value_range:
        total = 0.0

        # Replace the -2 in the vector with the current q_val to evaluate
        working_vector = query_vector.copy()
        working_vector[query_index] = q_val

        # Step 3: Enumerate over all hidden variables
        for s1 in range(2):
            for s2 in range(2):
                for s3 in range(2):
                    for s4 in range(2):
                        for cond in range(4):
                            full = [s1, s2, s3, s4, cond]
                            match = True
                            for i in range(5):
                                if query_vector[i] == -1:
                                    continue  # hidden, allow anything
                                if working_vector[i] != full[i]:
                                    match = False
                                    break
                            if match:
                                total += joint_probs[s1][s2][s3][s4][cond]
        
        result[q_val] = total

    # Step 4: Normalize the result
    total_sum = sum(result.values())
    if total_sum == 0:
        return {k: 0 for k in result}  

    for key in result:
        result[key] /= total_sum

    return result

def marginal_probabilities(joint: np.ndarray) -> list[dict[int, float]]:
    """
    Compute the marginal P(X_i) for each variable i in the joint distribution.
    Returns a list M of length n_vars, where
      M[i] is a dict mapping each possible value of X_i to P(X_i = value).
    """
    n_vars = joint.ndim
    marginals = []
    for i in range(n_vars):
        # build a query vector: -2 at i, -1 everywhere else
        qvec = [-1] * n_vars
        qvec[i] = -2
        post = inference_by_enumeration(joint, qvec)
        marginals.append(post)
    return marginals

