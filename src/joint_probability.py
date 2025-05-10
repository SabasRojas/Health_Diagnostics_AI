import csv
import numpy as np
import os

def load_data(csv_path):
    """
    Reads the CSV and returns a list of records.
    Each record is a list: [s1, s2, s3, s4, condition]
    """
    records = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            record = list(map(int, row))
            if len(record) == 5:
                records.append(record)
    return records


def build_joint_distribution(records, num_diseases):
    """
    Counts how many times each [s1][s2][s3][s4][condition] combination appears.
    Returns the raw count table and the normalized probability table.
    """
    joint_counts = np.zeros((2, 2, 2, 2, num_diseases), dtype=int)

    for record in records:
        s1, s2, s3, s4, cond = record
        joint_counts[s1][s2][s3][s4][cond] += 1

    total = np.sum(joint_counts)
    joint_probs = joint_counts / total if total > 0 else joint_counts
    return joint_counts, joint_probs

def save_distribution_to_file(prob_table, output_path):
    """
    Saves the non-zero probabilities to a text file for inspection.
    """
    with open(output_path, 'w') as f:
        for s1 in range(2):
            for s2 in range(2):
                for s3 in range(2):
                    for s4 in range(2):
                        for cond in range(4):
                            prob = prob_table[s1][s2][s3][s4][cond]
                            if prob > 0:
                                f.write(f"P([{s1},{s2},{s3},{s4},{cond}]) = {prob:.6f}\n")

def run_joint_distribution():
    """
    Full pipeline to load data, compute probabilities, and optionally save output.
    """
    csv_path = os.path.join("../data", "Health_Data_Set.csv")
    records = load_data(csv_path)
    counts, probs = build_joint_distribution(records, 4)
    save_distribution_to_file(probs, os.path.join("../data", "joint_distribution_output.txt"))
    return counts, probs

if __name__ == "__main__":
    run_joint_distribution()
