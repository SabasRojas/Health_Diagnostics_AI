import numpy as np
from src.inference import inference_by_enumeration

def test_simple_uniform_distribution():
    # Just making a uniform joint distribution over 2^4 binary vars and 1 4-class var
    joint = np.ones((2, 2, 2, 2, 4))
    joint /= joint.sum()  # Normalize (probably not necessary, but just in case)

    # Query: asking about the condition variable, no evidence given
    qvec = [-1, -1, -1, -1, -2]

    # Run inference and see what we get
    result = inference_by_enumeration(joint, qvec)

    print("Inference result (should be uniform):", result)  # DEBUG

    # Sanity check
    for val, prob in result.items():
        assert abs(prob - 0.25) < 1e-6, f"Expected ~0.25, got {prob} for value {val}"
