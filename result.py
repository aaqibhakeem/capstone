# result.py
import json
from kafka import KafkaConsumer

# Consumer listens to the "results" topic
consumer = KafkaConsumer(
    "results",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    group_id="results_aggregator",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    max_poll_records=1000
)

# Tracking per dataset
stats = {
    "ds1": {"correct": 0, "total": 0},
    "ds2": {"correct": 0, "total": 0},
    "ds3": {"correct": 0, "total": 0}
}

print("Result aggregator running (listening on 'results' topic)...")

for msg in consumer:
    res = msg.value
    ds = res.get("dataset")

    if ds not in stats:
        stats[ds] = {"correct": 0, "total": 0}

    # update stats from message
    stats[ds]["correct"] = res.get("correct", stats[ds]["correct"])
    stats[ds]["total"] = res.get("total", stats[ds]["total"])

    # compute per-dataset rates
    per_ds = {
        k: (v["correct"] / v["total"]) if v["total"] else 0.0
        for k, v in stats.items()
    }

    # compute overall
    overall_correct = sum(v["correct"] for v in stats.values())
    overall_total = sum(v["total"] for v in stats.values())
    overall_rate = (overall_correct / overall_total) if overall_total else 0.0

    # clean output
    print(
        f"Rates → "
        f"ds1={per_ds['ds1']:.4%}, "
        f"ds2={per_ds['ds2']:.4%}, "
        f"ds3={per_ds['ds3']:.4%} | "
        f"overall={overall_rate:.4%}"
    )
