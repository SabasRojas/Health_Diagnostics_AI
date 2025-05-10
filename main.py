import os
from src.joint_probability import load_data, build_joint_distribution
from src.inference import inference_by_enumeration, marginal_probabilities
from src.information_gain import expected_information_gain


def main():
    # Get path to the CSV file (adjust this if needed)
    simple_data_file = os.path.join(os.path.dirname(__file__), 'data', 'Health_Data_Set.csv')
    realistic_data_file = os.path.join(os.path.dirname(__file__), 'data', 'kaggle_clean_dataset.csv')

    # Load everything and set up the joint distribution
    simple_raw = load_data(simple_data_file)
    realistic_raw = load_data(realistic_data_file)
    simple_num_diseases = len(set([r[4] for r in simple_raw]))
    realistic_num_diseases = len(set([r[4] for r in realistic_raw]))
    simple_counts, simple_joint_dist = build_joint_distribution(simple_raw, simple_num_diseases)
    realistic_counts, realistic_joint_dist = build_joint_distribution(realistic_raw, realistic_num_diseases)
    
    print("Loaded data. Computing marginals...\n")
    print("SIMPLE DATASET RESULTS")
    print("="*20)
    # Compute marginals
    marginals = marginal_probabilities(simple_joint_dist)
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
        result = inference_by_enumeration(simple_joint_dist, q)
        for val, p in result.items():
            print(f"  val={val} --> {p:.4f}")
            
    print("\n=== Realistic Dataset Results (Kaggle) ===")      
            
            
    # "Realistic dataset inference"
    kaggle_disease_labels = {
        0: 'Influenza',
        1: 'Common Cold',
        2: 'Eczema',
        3: 'Asthma',
        4: 'Hyperthyroidism',
        5: 'Allergic Rhinitis',
        6: 'Anxiety Disorders',
        7: 'Diabetes',
        8: 'Gastroenteritis',
        9: 'Pancreatitis',
        10: 'Rheumatoid Arthritis',
        11: 'Depression',
        12: 'Liver Cancer',
        13: 'Stroke',
        14: 'Urinary Tract Infection',
        15: 'Dengue Fever',
        16: 'Hepatitis',
        17: 'Kidney Cancer',
        18: 'Migraine',
        19: 'Muscular Dystrophy',
        20: 'Sinusitis',
        21: 'Ulcerative Colitis',
        22: 'Bipolar Disorder',
        23: 'Bronchitis',
        24: 'Cerebral Palsy',
        25: 'Colorectal Cancer',
        26: 'Hypertensive Heart Disease',
        27: 'Multiple Sclerosis',
        28: 'Myocardial Infarction (Heart...',
        29: 'Urinary Tract Infection (UTI)',
        30: 'Osteoporosis',
        31: 'Pneumonia',
        32: 'Atherosclerosis',
        33: 'Chronic Obstructive Pulmonary...',
        34: 'Epilepsy',
        35: 'Hypertension',
        36: 'Obsessive-Compulsive Disorde...',
        37: 'Psoriasis',
        38: 'Rubella',
        39: 'Cirrhosis',
        40: 'Conjunctivitis (Pink Eye)',
        41: 'Liver Disease',
        42: 'Malaria',
        43: 'Spina Bifida',
        44: 'Kidney Disease',
        45: 'Osteoarthritis',
        46: 'Klinefelter Syndrome',
        47: 'Acne',
        48: 'Brain Tumor',
        49: 'Cystic Fibrosis',
        50: 'Glaucoma',
        51: 'Rabies',
        52: 'Chickenpox',
        53: 'Coronary Artery Disease',
        54: 'Eating Disorders (Anorexia,...',
        55: 'Fibromyalgia',
        56: 'Hemophilia',
        57: 'Hypoglycemia',
        58: 'Lymphoma',
        59: 'Tuberculosis',
        60: 'Lung Cancer',
        61: 'Hypothyroidism',
        62: 'Autism Spectrum Disorder (ASD)',
        63: "Crohn's Disease",
        64: 'Hyperglycemia',
        65: 'Melanoma',
        66: 'Ovarian Cancer',
        67: 'Turner Syndrome',
        68: 'Zika Virus',
        69: 'Cataracts',
        70: 'Pneumocystis Pneumonia (PCP)',
        71: 'Scoliosis',
        72: 'Sickle Cell Anemia',
        73: 'Tetanus',
        74: 'Anemia',
        75: 'Cholera',
        76: 'Endometriosis',
        77: 'Sepsis',
        78: 'Sleep Apnea',
        79: 'Down Syndrome',
        80: 'Ebola Virus',
        81: 'Lyme Disease',
        82: 'Pancreatic Cancer',
        83: 'Pneumothorax',
        84: 'Appendicitis',
        85: 'Esophageal Cancer',
        86: 'HIV/AIDS',
        87: 'Marfan Syndrome',
        88: "Parkinson's Disease",
        89: 'Hemorrhoids',
        90: 'Polycystic Ovary Syndrome (PCOS)',
        91: 'Systemic Lupus Erythematosus...',
        92: 'Typhoid Fever',
        93: 'Breast Cancer',
        94: 'Measles',
        95: 'Osteomyelitis',
        96: 'Polio',
        97: 'Chronic Kidney Disease',
        98: 'Hepatitis B',
        99: 'Prader-Willi Syndrome',
        100: 'Thyroid Cancer',
        101: 'Bladder Cancer',
        102: 'Otitis Media (Ear Infection)',
        103: 'Tourette Syndrome',
        104: "Alzheimer's Disease",
        105: 'Chronic Obstructive Pulmonary Disease (COPD)',
        106: 'Dementia',
        107: 'Diverticulitis',
        108: 'Mumps',
        109: 'Cholecystitis',
        110: 'Prostate Cancer',
        111: 'Schizophrenia',
        112: 'Gout',
        113: 'Testicular Cancer',
        114: 'Tonsillitis',
        115: 'Williams Syndrome'
    }
    
    # Inference w/ query
    query = [1, 1, -1, -1, -2]  # Fever = Yes, Cough = Yes, others unknown, querying disease
    result = inference_by_enumeration(realistic_joint_dist, query)
    
    # Results w/ disease names, sorted by probability 
    print("\n=== Inference Results ===")
    sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
    
    for disease_id, prob in sorted_results:
        if prob > 0:
            disease_name = kaggle_disease_labels.get(disease_id, "Unknown Disease")
            print(f"{disease_name} -> {round(prob, 4)}")

            
    gains = expected_information_gain(realistic_joint_dist, query)
    
    symptom_names = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    print("\n=== Value of Additional Information ===")
    for idx, gain in gains.items():
        print(f"Symptom '{symptom_names[idx]}' -> Expected Information Gain: {round(gain, 4)}")

if __name__ == "__main__":
    main()
