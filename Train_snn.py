import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

# Import hàm load từ file preprocess_audio.py
from Preprocess_audio import load_dataset

# 1. Cấu hình thiết bị
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Đang sử dụng thiết bị: {device}")

# 2. Load dữ liệu
# Thay đổi đường dẫn 'root' trong hàm load_dataset hoặc truyền vào đây
X, Y = load_dataset(root=r"E:\lab\SnnforAudio\data_voice")

if X is None:
    exit()

# Chuyển dữ liệu lên device
X, Y = X.to(device), Y.to(device)

# Tạo DataLoader để train theo batch (giúp train ổn định hơn)
batch_size = 8
train_data = TensorDataset(X, Y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# 3. Tham số SNN
beta = 0.9  # Tốc độ suy giảm điện thế màng (Decay rate)
# Surrogate gradient: Giúp đạo hàm truyền qua được xung (spike)
spike_grad = surrogate.fast_sigmoid(slope=25)

# 4. Định nghĩa mạng SNN
class AudioSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input features là 40 (số lượng MFCC)
        # Hidden layer 128 nơ-ron
        self.fc1 = nn.Linear(40, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Output layer 5 nơ-ron (tương ứng 5 class)
        self.fc2 = nn.Linear(128, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Input x đang có dạng: (Batch, Time, Features) -> (8, 50, 40)
        # SNN cần loop theo thời gian, nên ta đảo trục Time lên đầu
        # New shape: (Time, Batch, Features) -> (50, 8, 40)
        x = x.permute(1, 0, 2) 
        
        # Khởi tạo điện thế màng (membrane potential) tại t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # List để ghi lại các xung đầu ra (spikes) theo thời gian
        spk2_rec = []

        # Vòng lặp mô phỏng thời gian (Time-step simulation)
        for step in range(x.size(0)):
            # 1. Lớp ẩn
            cur1 = self.fc1(x[step])     # Xử lý input tại thời điểm 'step'
            spk1, mem1 = self.lif1(cur1, mem1) # Tích hợp và phát xung
            
            # 2. Lớp đầu ra
            cur2 = self.fc2(spk1)        # Input là xung của lớp trước
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Ghi lại xung đầu ra
            spk2_rec.append(spk2)

        # Stack lại thành Tensor: (Time, Batch, Output_Class)
        return torch.stack(spk2_rec, dim=0)

# 5. Khởi tạo Model, Loss, Optimizer
model = AudioSNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3) # Learning rate

print("\n--- Bắt đầu huấn luyện ---\n")
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Time, Batch, Classes)
        spk_rec = model(images)

        # Tính Loss: Rate Coding (Tổng số xung trên trục thời gian)
        # Sum theo dim 0 (Time) -> (Batch, Classes)
        loss_val = criterion(spk_rec.sum(dim=0), labels)

        # Backward pass
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()

        # Tính độ chính xác (Accuracy)
        # Lấy class có tổng số xung cao nhất
        _, predicted = spk_rec.sum(dim=0).max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

# 6. Lưu model
torch.save(model.state_dict(), "snn_audio_model.pt")
print("\nĐã lưu model thành công vào file 'snn_audio_model.pt'")