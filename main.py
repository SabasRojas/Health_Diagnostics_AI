import os
from src.joint_probability import load_data, build_joint_distribution
from src.inference import inference_by_enumeration, marginal_probabilities


def main():
    # Get path to the CSV file (adjust this if needed)
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'Health_Data_Set.csv')

    # Load everything and set up the joint distribution
    raw = load_data(data_file)
    counts, joint_dist = build_joint_distribution(raw)

    print("Loaded data. Computing marginals...\n")

    # Compute marginals
    marginals = marginal_probabilities(joint_dist)
    names = ['s1', 's2', 's3', 's4', 'Condition']

    print("=== Marginal Probabilities ===")
    for idx, m in enumerate(marginals):
        print(f"{names[idx]}:")
        for val, prob in sorted(m.items()):
            print(f"  {names[idx]}={val} -> {round(prob, 4)}")
        print()

    # Just for quick sanity check
    print("First few condition marginals:")
    print(marginals[4])

    # Try out a few queries
    print("\n=== Inference Examples ===")
    test_cases = [
        ([1, -1, -1, -1, -2], "P(C | s1=1)"),
        ([0, 1, -1, 0, -2], "P(C | s1=0, s2=1, s4=0)"),
        ([-2, 1, 0, -1, -1], "P(s1 | s2=1, s3=0)"),
        ([-1, -1, -1, 1, -2], "P(C | s4=1)"),
    ]

    for q, label in test_cases:
        print(f"\n{label}:")
        result = inference_by_enumeration(joint_dist, q)
        for val, p in result.items():
            print(f"  val={val} --> {p:.4f}")

if __name__ == "__main__":
    main()
