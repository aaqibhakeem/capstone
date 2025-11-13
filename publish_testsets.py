# publish_testsets.py
import random
import sys

DATASETS = {
    "ds1": "./honeynet.txt",
    "ds2": "./000webhost.txt",
    "ds3": "./myspace.txt",
}

def main(dataset_key):
    if dataset_key not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_key}. Choose from {list(DATASETS.keys())}")
        return

    src_file = DATASETS[dataset_key]
    with open(src_file, "r", encoding="utf-8", errors="ignore") as f:
        passwords = f.read().splitlines()

    testset = random.sample(passwords, min(10000, len(passwords)))

    out_file = f"testset_{dataset_key}.txt"
    with open(out_file, "w", encoding="utf-8") as out_f:
        for pwd in testset:
            out_f.write(pwd + "\n")

    print(f"Saved {out_file} with {len(testset)} entries")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python publish_testsets.py <ds1|ds2|ds3>")
        sys.exit(1)
    main(sys.argv[1])

