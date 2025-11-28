# predict.py
import argparse
import json
import math
import torch
import torch.nn as nn
from kafka import KafkaProducer

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PGTCN(nn.Module):
    def __init__(self, vocab_size, emb_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = emb_size if i == 0 else num_channels[i-1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=(kernel_size-1)*dilation, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], vocab_size)
    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        tcn_out = self.tcn(emb).transpose(1, 2)
        return self.fc(tcn_out)

# ---------- utilities for robust checkpoint loading ----------
def build_shape_index(state_dict):
    idx = {}
    for k, v in state_dict.items():
        try:
            shape = tuple(v.shape)
        except Exception:
            continue
        idx.setdefault(shape, []).append(k)
    return idx

def map_by_shape(model_state, checkpoint_state):
    from difflib import SequenceMatcher
    ck_shape_index = build_shape_index(checkpoint_state)
    used_ck_keys = set()
    mapped = {}
    collisions = []
    for m_key, m_tensor in model_state.items():
        m_shape = tuple(m_tensor.shape)
        candidates = ck_shape_index.get(m_shape, [])
        chosen = None
        for ckey in candidates:
            if ckey in used_ck_keys:
                continue
            if m_key in ckey or ckey in m_key:
                chosen = ckey
                break
        if chosen is None and candidates:
            best_score = 0.0
            best_key = None
            for ckey in candidates:
                if ckey in used_ck_keys:
                    continue
                score = SequenceMatcher(None, m_key, ckey).ratio()
                if score > best_score:
                    best_score = score
                    best_key = ckey
            if best_key is not None:
                chosen = best_key
        if chosen is None and candidates:
            for ckey in candidates:
                if ckey not in used_ck_keys:
                    chosen = ckey
                    break
        if chosen is not None:
            mapped[m_key] = checkpoint_state[chosen]
            used_ck_keys.add(chosen)
        else:
            collisions.append(m_key)
    return mapped, collisions, used_ck_keys

def quick_name_remap(raw_state):
    remapped = {}
    for k, v in raw_state.items():
        if k.startswith("layer_norm"):
            continue
        if "tcn.network." in k:
            rest = k.split("tcn.network.", 1)[1]
            parts = rest.split(".")
            if len(parts) >= 3:
                idx = parts[0]
                conv = parts[1]
                param = ".".join(parts[2:])
                try:
                    idx_int = int(idx)
                except Exception:
                    idx_int = idx
                if conv == "conv1":
                    new_key = f"tcn.{idx_int}.net.0.{param}"
                elif conv == "conv2":
                    new_key = f"tcn.{idx_int}.net.4.{param}"
                else:
                    new_key = k
                remapped[new_key] = v
                continue
        if k.startswith("tcn."):
            parts = k.split(".")
            if len(parts) >= 4 and parts[0] == "tcn":
                idx = parts[1]
                conv = parts[2]
                param = ".".join(parts[3:])
                try:
                    idx_int = int(idx)
                except Exception:
                    idx_int = idx
                if conv == "conv1":
                    new_key = f"tcn.{idx_int}.net.0.{param}"
                    remapped[new_key] = v
                    continue
                elif conv == "conv2":
                    new_key = f"tcn.{idx_int}.net.4.{param}"
                    remapped[new_key] = v
                    continue
        remapped[k] = v
    return remapped

def load_and_map_checkpoint(model, model_path, device):
    raw = torch.load(model_path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw_state = raw["state_dict"]
    else:
        raw_state = raw
    ck_tensors = {k: v for k, v in raw_state.items() if hasattr(v, "shape")}
    quick_mapped_ck = quick_name_remap(ck_tensors)
    model_state = dict(model.state_dict())
    mapped_state, collisions, used_ck_keys = map_by_shape(model_state, quick_mapped_ck)
    if collisions:
        fallback_mapped, collisions2, used2 = map_by_shape({k: model_state[k] for k in collisions}, ck_tensors)
        for k, v in fallback_mapped.items():
            mapped_state[k] = v
        collisions = collisions2
    final_state = {}
    for k in model_state.keys():
        if k in mapped_state:
            final_state[k] = mapped_state[k]
    load_res = model.load_state_dict(final_state, strict=False)
    try:
        missing_keys = list(load_res.missing_keys)
        unexpected_keys = list(load_res.unexpected_keys)
    except Exception:
        try:
            missing_keys, unexpected_keys = load_res
        except Exception:
            missing_keys = [k for k in model_state.keys() if k not in final_state]
            unexpected_keys = []
    return missing_keys, unexpected_keys

# ---------- beam search (token-id based) ----------
def beam_search_single(model, device, start_idx, eos_idx, beam_width=5, max_len=20):
    """
    Returns a list of token-id lists for the best beam (top result only).
    We run beam search and return the top hypothesis (as list of ints).
    """
    model.eval()
    # initial input: batch size = 1, sequence length = 1 (start token)
    input_ids = torch.tensor([[start_idx]], dtype=torch.long, device=device)

    # beam entries: (sequence_tensor, logprob_sum, ended_flag)
    beams = [(input_ids, 0.0, False)]

    with torch.no_grad():
        for step in range(max_len):
            all_candidates = []
            for seq_tensor, score, ended in beams:
                if ended:
                    # carry forward finished beam unchanged
                    all_candidates.append((seq_tensor, score, True))
                    continue

                # model expects shape (batch, seq_len)
                logits = model(seq_tensor)  # shape (1, seq_len, vocab)
                last_logits = logits[:, -1, :].squeeze(0)  # (vocab,)
                log_probs = torch.log_softmax(last_logits, dim=-1).cpu()

                # pick top-k candidates from this beam (k = beam_width)
                topk_vals, topk_indices = torch.topk(log_probs, min(beam_width, log_probs.size(0)))
                for v, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
                    new_seq = torch.cat([seq_tensor, torch.tensor([[idx]], device=device)], dim=1)
                    new_score = score + float(v)  # accumulate log-probs
                    is_end = (idx == eos_idx)
                    all_candidates.append((new_seq, new_score, is_end))

            # keep top beam_width candidates overall
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # if every beam has ended, stop early
            if all(b[2] for b in beams):
                break

    # choose the top beam (highest score)
    best_seq_tensor, best_score, ended_flag = beams[0]
    # convert tensor to python list of ints (skip the start token optionally)
    ids = best_seq_tensor.squeeze(0).cpu().tolist()  # includes start_idx and possibly eos
    return ids

# ---------- main ----------
def main(dataset, count, model_path, cfg_path, kafka_bootstrap="localhost:9092",
         batch_size=200, device="cpu", beam_width=5, max_len=20, start_idx=1, eos_idx=2):
    # load config only for model params (vocab_size, channels, etc.)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = PGTCN(vocab_size=cfg["vocab_size"],
                  emb_size=cfg["embedding_dim"],
                  num_channels=cfg["tcn_channels"],
                  kernel_size=cfg["kernel_size"],
                  dropout=cfg["dropout"])

    missing_keys, unexpected_keys = load_and_map_checkpoint(model, model_path, device)
    model.to(device).eval()

    print("Model load summary:")
    if missing_keys:
        print("Missing keys (expected by model but absent in checkpoint):")
        for k in missing_keys:
            print(" ", k)
    else:
        print("No missing keys.")
    if unexpected_keys:
        print("Unexpected keys (present in checkpoint but not used by model):")
        for k in unexpected_keys[:200]:
            print(" ", k)
        if len(unexpected_keys) > 200:
            print(f" ... and {len(unexpected_keys)-200} more unexpected keys")
    else:
        print("No unexpected keys.")

    # Kafka producer
    producer = KafkaProducer(bootstrap_servers=kafka_bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode("utf-8"))
    topic = f"guesses_{dataset}"
    out_path = f"guesses_{dataset}.txt"
    buffer = []

    print(f"Generating {count} guesses for {dataset} with beam_width={beam_width}, max_len={max_len}...")

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i in range(count):
            if i % 100 == 0:
                print(f"Generated {i}/{count} guesses...", end="\r", flush=True)

            ids = beam_search_single(model, device, start_idx=start_idx, eos_idx=eos_idx,
                                     beam_width=beam_width, max_len=max_len)

            # Format: write space-separated token ids (skip start token if present)
            # If start token is at position 0, we will drop it for the saved guess to match expectation.
            if len(ids) > 0 and ids[0] == start_idx:
                save_ids = ids[1:]
            else:
                save_ids = ids

            out_line = " ".join(str(x) for x in save_ids)
            out_f.write(out_line + "\n")

            msg = {"dataset": dataset, "guess_ids": save_ids}
            buffer.append(msg)

            if len(buffer) >= batch_size:
                for m in buffer:
                    producer.send(topic, m)
                producer.flush()
                buffer.clear()

        # flush remainder
        if buffer:
            for m in buffer:
                producer.send(topic, m)
            producer.flush()
            buffer.clear()

    print(f"\nFinished generating {count} guesses for {dataset}.")
    producer.close()
    print(f"Saved guesses to {out_path} and published to topic {topic}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds1", "ds2", "ds3"])
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--model", default="pgtcn_best_myspace.pt", help="Path to model checkpoint")
    parser.add_argument("--cfg", default="config_myspace.json", help="Model config (contains vocab_size, channels, etc.)")
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--max-len", type=int, default=20, help="Max generation length (including or excluding start token depending on start_idx)")
    parser.add_argument("--start-idx", type=int, default=1, help="Start token index (used to prime generation)")
    parser.add_argument("--eos-idx", type=int, default=2, help="EOS token index (generation stops when produced)")
    args = parser.parse_args()

    main(args.dataset, args.count, args.model, args.cfg, args.kafka,
         args.batch_size, args.device, args.beam, args.max_len, args.start_idx, args.eos_idx)
