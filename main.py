from src.joint_probability import run_joint_distribution
from src.inference import inference_by_enumeration

def main():
    print("Building joint probability distribution from health data...")
    counts, joint_probs = run_joint_distribution()
    print("Done!")
    print(f"Total records processed: {counts.sum()}")
    print("Non-zero probabilities saved to: data/joint_distribution_output.txt\n")

    # Example: We're asking "What's the probability of each condition if the patient has symptom1=1 and symptom4=0?"
    # -2 = query (condition), -1 = hidden, 0/1/2/3 = known value
    query_vector = [1, -1, -1, 0, -2]  # symptoms 1 and 4 known, rest unknown; condition is query

    print("Running inference on:")
    print("  Symptom 1 = 1")
    print("  Symptom 4 = 0")
    print("  Others unknown")
    print("  Query: Condition")
    result = inference_by_enumeration(joint_probs, query_vector)
    print("\nResulting probabilities for each condition:")
    for cond, prob in result.items():
        print(f"  Condition {cond}: {prob:.4f}")

if __name__ == "__main__":
    main()
