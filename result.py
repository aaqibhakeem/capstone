import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "results",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    group_id="aggregator",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

print("📊 Results Aggregator running...")

# Track cumulative stats
stats = {
    "ds1": {"matches": 0, "total": 0},
    "ds2": {"matches": 0, "total": 0},
    "ds3": {"matches": 0, "total": 0},
}

for msg in consumer:
    result = msg.value
    ds = result["dataset"]

    stats[ds]["matches"] += result["matches"]
    stats[ds]["total"] += result["total"]

    # Calculate per-dataset and overall rates
    match_rate_ds = stats[ds]["matches"] / stats[ds]["total"] if stats[ds]["total"] > 0 else 0.0

    overall_matches = sum(s["matches"] for s in stats.values())
    overall_total = sum(s["total"] for s in stats.values())
    overall_rate = overall_matches / overall_total if overall_total > 0 else 0.0

    print(f"📌 Dataset {ds} → Match rate: {match_rate_ds:.4%}")
