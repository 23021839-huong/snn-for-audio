import os
import librosa
import numpy as np
import torch

def load_dataset(root=r"E:\lab\SnnforAudio\data_voice", max_len=50, n_mfcc=40):
    """
    Load dataset âm thanh và chuyển đổi sang MFCC.
    Output Shape: (Batch_Size, Time_Steps, Features) -> (N, 50, 40)
    """
    X, Y = [], []

    labels = ["1", "2", "3", "4", "5"]

    print("Đang xử lý dữ liệu...")
    
    for label in labels:
        folder = os.path.join(root, label)

        if not os.path.exists(folder):
            print(f"⚠ Cảnh báo: Thư mục không tồn tại: {folder}")
            continue

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)

                # 1. Load file âm thanh
                # sr=16000 là chuẩn phổ biến cho nhận dạng giọng nói
                try:
                    audio, sr = librosa.load(path, sr=16000)
                except Exception as e:
                    print(f"Lỗi đọc file {file}: {e}")
                    continue

                # 2. Trích xuất đặc trưng MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
                
                # 3. Chuẩn hóa (Normalization) để đưa giá trị về khoảng nhỏ
                mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-9)

                # 4. Padding hoặc Cắt (Truncate) để có độ dài thời gian cố định (max_len)
                # mfcc shape hiện tại: (n_mfcc, time) -> (40, biến thiên)
                current_len = mfcc.shape[1]
                
                if current_len < max_len:
                    pad_width = max_len - current_len
                    pad = np.zeros((n_mfcc, pad_width))
                    mfcc = np.concatenate([mfcc, pad], axis=1)
                else:
                    mfcc = mfcc[:, :max_len]

                # 5. QTranspose (Hoán đổi chiều)
                # Chuyển từ (Features, Time) -> (Time, Features)
                # Kết quả: (50, 40)
                mfcc = mfcc.T

                X.append(mfcc)
                # Label: "1" -> 0, "2" -> 1...
                Y.append(int(label) - 1)

    if len(X) == 0:
        print("Lỗi: Không tìm thấy dữ liệu nào!")
        return None, None

    # Chuyển sang Tensor của PyTorch
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.long)

    print("Dataset loaded thành công!")
    print(f"X Shape: {X.shape} (Batch, Time, Features)")
    print(f"Y Shape: {Y.shape}")

    return X, Y

if __name__ == "__main__":
    # Test thử hàm load
    load_dataset()
    