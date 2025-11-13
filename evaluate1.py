import sys, json
import os
from kafka import KafkaConsumer, KafkaProducer

def normalize(pwd: str) -> str:
    """Strip whitespace + lowercase for consistent comparison"""
    return pwd.strip().lower()

def run_evaluator(dataset_id):
    #############################################
    # Config
    #############################################
    TESTSET_PATH = f"./testset_{dataset_id}.txt"
    GUESSES_TOPIC = f"guesses_{dataset_id}"
    RESULTS_TOPIC = "results"

    #############################################
    # Load test set into memory
    #############################################
    if not os.path.exists(TESTSET_PATH):
        print(f"❌ Test set file not found: {TESTSET_PATH}")
        return

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        test_passwords = set(normalize(line) for line in f if line.strip())

    print(f"✅ Loaded {len(test_passwords)} passwords for {dataset_id}")

    #############################################
    # Kafka setup
    #############################################
    consumer = KafkaConsumer(
        GUESSES_TOPIC,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        group_id=f"evaluator_{dataset_id}"
    )
    producer = KafkaProducer(bootstrap_servers="localhost:9092")

    #############################################
    # Evaluation loop
    #############################################
    total_guesses = 0
    matches = 0

    print(f"🔍 Evaluator for {dataset_id} running...")

    for msg in consumer:
        guess = normalize(msg.value.decode("utf-8"))
        total_guesses += 1
        if guess in test_passwords:
            matches += 1
            # Show first few matches for debugging
            if matches <= 5:
                print(f"🎯 MATCH FOUND: {guess}")

        match_rate = matches / total_guesses if total_guesses > 0 else 0.0

        # Publish partial result
        result_msg = {
            "dataset": dataset_id,
            "total_guesses": total_guesses,
            "matches": matches,
            "match_rate": round(match_rate, 4)
        }
        producer.send(RESULTS_TOPIC, value=json.dumps(result_msg).encode("utf-8"))

        if total_guesses % 100 == 0:
            print(f"[{dataset_id}] guesses={total_guesses}, matches={matches}, rate={match_rate:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <ds1|ds2|ds3>")
        sys.exit(1)

    dataset_id = sys.argv[1]
    run_evaluator(dataset_id)
