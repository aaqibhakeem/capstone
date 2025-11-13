import sys
import json
import torch
import torch.nn.functional as F
from kafka import KafkaProducer
import pickle
import random
import torch.nn as nn

#############################################
# Model definition (same PGTCN as training)
#############################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
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
    def __init__(self, vocab_size, emb_size=128, num_channels=[128]*3, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = emb_size if i == 0 else num_channels[i-1]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation, padding=(kernel_size-1)*dilation)
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], vocab_size)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        tcn_out = self.tcn(emb).transpose(1, 2)
        return self.fc(tcn_out)

#############################################
# Utils (sampling)
#############################################
def sample_from_logits(logits, temperature=1.0, top_k=20):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values / temperature, dim=-1)
        idx = indices[torch.multinomial(probs, 1)]
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1)
    return idx.item()

def generate_password(model, stoi, itos, max_len=20, temperature=0.7, start_char="<SOS>"):
    model.eval()
    device = next(model.parameters()).device

    if start_char in stoi:
        x = torch.tensor([[stoi[start_char]]], dtype=torch.long, device=device)
    else:
        x = torch.tensor([[random.randint(0, len(stoi)-1)]], dtype=torch.long, device=device)

    pwd = ""
    for _ in range(max_len):
        logits = model(x)[:, -1, :]
        idx = sample_from_logits(logits[-1], temperature=temperature, top_k=20)
        char = itos[idx]
        if char == "<EOS>":
            break
        pwd += char
        x = torch.cat([x, torch.tensor([[idx]], device=device)], dim=1)
    return pwd

#############################################
# Main
#############################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py [ds1|ds2|ds3]")
        sys.exit(1)

    dataset = sys.argv[1]
    guesses_topic = f"guesses_{dataset}"

    # Load vocab
    with open("stoi.pkl", "rb") as f:
        stoi = pickle.load(f)
    with open("itos.pkl", "rb") as f:
        itos = pickle.load(f)

    vocab_size = len(stoi)

    # Rebuild model and load weights
    model = PGTCN(vocab_size)
    state_dict = torch.load("pgtcn_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    total_guesses = 3000
    guesses = []

    print(f"🔮 Generating {total_guesses} guesses for {dataset}...")

    for _ in range(total_guesses):
        pwd = generate_password(model, stoi, itos, max_len=20)
        guesses.append(pwd)
        producer.send(guesses_topic, pwd)

    producer.flush()
    print(f"✅ Published {len(guesses)} guesses to Kafka topic: {guesses_topic}")

    # Save locally
    out_file = f"guesses_{dataset}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for g in guesses:
            f.write(g + "\n")

    print("🎉 Prediction complete. Exiting.")

