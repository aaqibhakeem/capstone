# evaluate.py
import argparse
import json
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime, timezone, timedelta

ISO_TZ = timezone(timedelta(hours=5, minutes=30))  # IST

def load_testset(path):
    """Load testset lines into a lowercase set."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Testset file not found: {path}")
        return set()

def publish_result(producer, dataset, correct, total, done):
    rate = correct / total if total else 0.0
    msg = {
        "dataset": dataset,
        "correct": correct,
        "total": total,
        "rate": rate,
        "done": done,
        "ts": datetime.now(ISO_TZ).isoformat(timespec="seconds")
    }
    producer.send("results", msg)
    producer.flush()

def evaluate(dataset, kafka_bootstrap="localhost:9092"):
    test_file = f"testset_{dataset}.txt"
    gt = load_testset(test_file)

    print(f"Evaluating guesses for {dataset}...")

    consumer = KafkaConsumer(
        f"guesses_{dataset}",
        bootstrap_servers=kafka_bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=5000,
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )

    producer = KafkaProducer(
        bootstrap_servers=kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    seen = set()
    correct = 0
    total = 0

    for msg in consumer:
        val = msg.value
        pwd = val.get("guess", "").strip().lower()

        # Only count unique guesses
        if pwd not in seen:
            seen.add(pwd)
            total += 1

            if pwd in gt:
                correct += 1

            # Publish incremental update
            publish_result(producer, dataset, correct, total, False)

    # Final snapshot
    publish_result(producer, dataset, correct, total, True)

    print(f"Evaluation finished for {dataset}: {correct}/{total} matched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate matching rate for guesses.")
    parser.add_argument("--dataset", required=True, choices=["ds1", "ds2", "ds3"])
    parser.add_argument("--kafka", default="localhost:9092")
    args = parser.parse_args()

    evaluate(args.dataset, args.kafka)
