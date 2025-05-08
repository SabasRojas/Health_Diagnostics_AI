import os
import numpy as np
import pytest

from src.joint_probability import load_data, build_joint_distribution
from src.inference import inference_by_enumeration, marginal_probabilities

# Define the relative path to the dataset
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Health_Data_Set.csv')

@pytest.fixture(scope="module")
def joint_probs():
    """
    Fixture that loads health data from a CSV file and builds a joint probability distribution.
    Runs once per module to avoid redundant computation.
    """
    raw_data = load_data(CSV_PATH)
    _, joint_distribution = build_joint_distribution(raw_data)
    return joint_distribution

def test_marginals_of_uniform_distribution():
    """
    Checks that a uniform joint distribution produces expected marginals.
    Binary variables should have equal probabilities; 4-class variables should split evenly.
    """
    # Create a joint distribution where every state is equally likely
    joint = np.ones((2, 2, 2, 2, 4))
    joint /= joint.sum()

    marginals = marginal_probabilities(joint)

    # Test binary symptom variables (indices 0–3)
    for i in range(4):
        prob_0 = marginals[i][0]
        prob_1 = marginals[i][1]
        assert pytest.approx(prob_0 + prob_1, rel=1e-8) == 1.0
        assert pytest.approx(prob_0, rel=1e-8) == 0.5

    # Test 4-class disease variable (index 4)
    condition_probs = marginals[4]
    total = sum(condition_probs.values())
    assert pytest.approx(total, rel=1e-8) == 1.0
    for p in condition_probs.values():
        assert pytest.approx(p, rel=1e-8) == 0.25

@pytest.mark.parametrize("query, description", [
    ([1, -1, -1, -1, -2], "P(C | s1=1)"),
    ([0, 1, -1, 0, -2], "P(C | s1=0, s2=1, s4=0)"),
    ([-2, 1, 0, -1, -1], "P(s1 | s2=1, s3=0)"),
    ([-1, -1, -1, 1, -2], "P(C | s4=1)"),
])
def test_inference_output_is_normalized(joint_probs, query, description):
    """
    Ensures that the output of the inference function is a valid probability distribution.
    """
    result = inference_by_enumeration(joint_probs, query)
    total_prob = sum(result.values())
    assert pytest.approx(total_prob, rel=1e-8) == 1.0, f"Distribution for {description} doesn't sum to 1"

    # Uncomment to debug or inspect output:
    # print(f"{description} => {result}")
