# publish_testsets.py
import argparse
import json
import os
from kafka import KafkaProducer

def split_and_publish(src_path: str, out_dir: str = ".", kafka_bootstrap="localhost:9092",
                      batch_size: int = 500):
    os.makedirs(out_dir, exist_ok=True)
    producers = {
        "ds1": KafkaProducer(bootstrap_servers=kafka_bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode("utf-8")),
        "ds2": KafkaProducer(bootstrap_servers=kafka_bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode("utf-8")),
        "ds3": KafkaProducer(bootstrap_servers=kafka_bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode("utf-8")),
    }

    outs = {
        "ds1": open(os.path.join(out_dir, "testset_ds1.txt"), "w", encoding="utf-8"),
        "ds2": open(os.path.join(out_dir, "testset_ds2.txt"), "w", encoding="utf-8"),
        "ds3": open(os.path.join(out_dir, "testset_ds3.txt"), "w", encoding="utf-8"),
    }

    buffers = {"ds1": [], "ds2": [], "ds3": []}
    topic_map = {"ds1": "testset_ds1", "ds2": "testset_ds2", "ds3": "testset_ds3"}

    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            pwd = line.rstrip("\n")
            if not pwd:
                continue
            # Round-robin assignment for even distribution and single-pass processing
            ds_key = ["ds1", "ds2", "ds3"][idx % 3]
            outs[ds_key].write(pwd + "\n")

            msg = {"dataset": ds_key, "password": pwd}
            buffers[ds_key].append(msg)

            if len(buffers[ds_key]) >= batch_size:
                for m in buffers[ds_key]:
                    producers[ds_key].send(topic_map[ds_key], m)
                producers[ds_key].flush()
                buffers[ds_key].clear()

    # flush remaining buffers and close files/producers
    for ds_key in ("ds1", "ds2", "ds3"):
        if buffers[ds_key]:
            for m in buffers[ds_key]:
                producers[ds_key].send(topic_map[ds_key], m)
            producers[ds_key].flush()
            buffers[ds_key].clear()
        outs[ds_key].close()
        producers[ds_key].close()

    print(f"Saved testset_ds1.txt, testset_ds2.txt, testset_ds3.txt and published to Kafka topics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split myspace.txt into 3 testsets and publish to Kafka.")
    parser.add_argument("--src", required=True, help="Path to myspace.txt (source file)")
    parser.add_argument("--out-dir", default=".", help="Directory to write testset files")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--batch-size", type=int, default=500, help="Kafka send batch size")
    args = parser.parse_args()
    split_and_publish(args.src, args.out_dir, args.kafka, args.batch_size)
