import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from Preprocess_audio import load_dataset

# 1. DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. LOAD DATA (1–9)
X, Y = load_dataset(root=r"E:\lab\SnnforAudio\data_voice")

if X is None:
    raise RuntimeError("Dataset empty!")

X, Y = X.to(device), Y.to(device)

dataset = TensorDataset(X, Y)

# Split 80% train – 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

batch_size = 8
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# 3. SNN PARAMETERS
beta = 0.9
spike_grad = surrogate.fast_sigmoid(slope=25)
# 4. MODEL
class AudioSNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(40, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, 9)  # 9 classes (1–9)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x: (Batch, Time, Features)
        x = x.permute(1, 0, 2)  # (Time, Batch, Features)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out = []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        return torch.stack(spk_out, dim=0)  # (Time, Batch, 9)

# 5. TRAIN SETUP
model = AudioSNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

epochs = 30
print("\nStart training...\n")

# 6. TRAIN + VALIDATION
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=False
    )

    for inputs, labels in train_bar:
        optimizer.zero_grad()

        spk_rec = model(inputs)
        out = spk_rec.sum(dim=0)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = out.max(1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_bar.set_postfix(
            accuracy=f"{train_correct/train_total:.4f}",
            loss=f"{train_loss/(train_total//labels.size(0)+1):.4f}"
        )

    train_acc = train_correct / train_total
    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            spk_rec = model(inputs)
            out = spk_rec.sum(dim=0)

            loss = criterion(out, labels)
            val_loss += loss.item()

            _, predicted = out.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    # LOG (KERAS STYLE) 
    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"accuracy: {train_acc:.4f} - loss: {train_loss:.4f} - "
        f"val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}"
    )

# 7. SAVE MODEL
torch.save(model.state_dict(), "snn_audio_model.pt")
print("\nModel saved as snn_audio_model.pt")
