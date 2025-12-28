import os
import librosa
import numpy as np
import torch


def load_dataset(root=r"E:\lab\SnnforAudio\data_voice",
                 max_len=50,
                 n_mfcc=40):
    """
    Load audio dataset và chuyển đổi sang MFCC.

    Output:
        X: Tensor (N, Time=50, Features=40)
        Y: Tensor (N,)
    """

    X, Y = [], []

    # Nhận diện chữ số từ 1 đến 9
    labels = [str(i) for i in range(1, 10)]

    print("Đang xử lý dữ liệu...")

    for label in labels:
        folder = os.path.join(root, label)

        if not os.path.exists(folder):
            print(f"Thư mục không tồn tại: {folder}")
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".wav"):
                continue

            path = os.path.join(folder, file)

            try:
                # 1. Load audio với sampling rate 16kHz
                audio, sr = librosa.load(path, sr=16000)
            except Exception as e:
                print(f"Lỗi đọc file {path}: {e}")
                continue

            # 2. Trích xuất MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc
            )

            # 3. Chuẩn hóa biên độ
            mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-9)

            # 4. Chuẩn hóa chiều thời gian
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                pad = np.zeros((n_mfcc, pad_width))
                mfcc = np.concatenate((mfcc, pad), axis=1)
            else:
                mfcc = mfcc[:, :max_len]

            # 5. Đổi trục (Features, Time) -> (Time, Features)
            mfcc = mfcc.T  # (50, 40)

            X.append(mfcc)
            Y.append(int(label) - 1)  # nhãn 0–8

    if len(X) == 0:
        print("Không tìm thấy dữ liệu hợp lệ!")
        return None, None

    # Chuyển sang Tensor
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.long)

    print("Dataset loaded thành công.")
    print(f"X shape: {X.shape} (Batch, Time, Features)")
    print(f"Y shape: {Y.shape}")

    return X, Y


if __name__ == "__main__":
    load_dataset()
