import sys
import os
import time

def normalize(pwd: str) -> str:
    return pwd.strip().lower()

def run_file_evaluator(dataset_id):
    #############################################
    # Config
    #############################################
    TESTSET_PATH = f"./testset_{dataset_id}.txt"
    GUESSES_PATH = f"./guesses_{dataset_id}.txt"

    #############################################
    # Load test set
    #############################################
    if not os.path.exists(TESTSET_PATH):
        print(f"❌ Test set file not found: {TESTSET_PATH}")
        return
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        test_passwords = set(normalize(line) for line in f if line.strip())
    print(f"✅ Loaded {len(test_passwords)} passwords for {dataset_id}")

    #############################################
    # Load guesses
    #############################################
    if not os.path.exists(GUESSES_PATH):
        print(f"❌ Guesses file not found: {GUESSES_PATH}")
        return
    with open(GUESSES_PATH, "r", encoding="utf-8") as f:
        guesses = [normalize(line) for line in f if line.strip()]
    print(f"✅ Loaded {len(guesses)} guesses for {dataset_id}")

    #############################################
    # Simulated Kafka-style evaluation loop
    #############################################
    total_guesses = 0
    matches = 0

    print(f"🔍 Evaluator for {dataset_id} running...")

    for guess in guesses:
        total_guesses += 1
        if guess in test_passwords:
            matches += 1

        if total_guesses % 100 == 0:
            match_rate = matches / total_guesses
            print(f"[{dataset_id}] guesses={total_guesses}, matches={matches}, rate={match_rate:.4f}")

        # tiny delay to simulate streaming
        time.sleep(0.001)

    #############################################
    # Final result
    #############################################
    match_rate = matches / total_guesses if total_guesses > 0 else 0.0
    print(f"\n✅ Final results for {dataset_id}: guesses={total_guesses}, matches={matches}, rate={match_rate:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <ds1|ds2|ds3>")
        sys.exit(1)

    dataset_id = sys.argv[1]
    run_file_evaluator(dataset_id)
