import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate



#     1. ENERGY-BASED VAD
def vad_energy(audio, sr, frame_ms=30, threshold=0.0015):
    """
    Cắt bỏ im lặng dựa trên năng lượng tín hiệu.
    threshold càng nhỏ → VAD càng nhạy.
    """

    frame_len = int(sr * frame_ms / 1000)
    energies = []

    # tính năng lượng từng frame
    for i in range(0, len(audio), frame_len):
        frame = audio[i:i+frame_len]
        energy = np.sum(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)

    speech_frames = energies > threshold
    speech_idx = np.where(speech_frames)[0]

    if len(speech_idx) == 0:
        print("⚠ Không phát hiện tiếng nói rõ, dùng toàn bộ audio.")
        return audio

    start = speech_idx[0] * frame_len
    end = (speech_idx[-1] + 1) * frame_len

    return audio[start:end]


#     2. RECORD FUNCTION
def record_to_file(out_path="record_test.wav", duration=1.5, sr=16000):
    print(f"🎤 Đang ghi âm {duration}s... Hãy nói số 1–5.")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    print("🔍 Đang tách tiếng nói (Energy VAD)...")
    trimmed = vad_energy(audio, sr)

    if len(trimmed) < 3000:
        print("⚠ Tiếng nói quá ngắn, dùng toàn bộ bản ghi.")
        trimmed = audio

    sf.write(out_path, trimmed, sr)
    print("📁 Đã lưu file:", out_path)

    return out_path



#     3. MFCC PROCESSING

def preprocess_file(path, max_len=50, n_mfcc=40):
    audio, sr = librosa.load(path, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-9)

    if mfcc.shape[1] < max_len:
        pad = np.zeros((n_mfcc, max_len - mfcc.shape[1]))
        mfcc = np.concatenate([mfcc, pad], axis=1)
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc.T  # → (Time=50, Features=40)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # thêm batch dim



#     4. AUDIO SNN MODEL

class AudioSNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(40, 128)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25))

        self.fc2 = nn.Linear(128, 5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25))

    def forward(self, x):
        # x: (Batch=1, Time=50, Features=40)
        x = x.permute(1, 0, 2)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0)  # (Time, Batch, 5)



#     5. PREDICT

def predict(model, mfcc):
    with torch.no_grad():
        out = model(mfcc)       # (Time, Batch, Classes)
        summed = out.sum(dim=0) # (Batch, Classes)

        probs = torch.softmax(summed, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        return pred + 1, probs.cpu().numpy()



#     6. MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--duration", type=float, default=1.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AudioSNN().to(device)
    model.load_state_dict(torch.load("snn_audio_model.pt", map_location=device))
    model.eval()

    # Xử lý input
    if args.record:
        input_path = record_to_file(duration=args.duration)
    elif args.file:
        input_path = args.file
    else:
        print("Hãy chọn --record hoặc --file <path>")
        return

    mfcc = preprocess_file(input_path).to(device)

    pred, probs = predict(model, mfcc)

    print("\n===== KẾT QUẢ =====")
    print("File:", input_path)
    print("Dự đoán:", pred)
    print("Softmax:", probs)

#     RUN

if __name__ == "__main__":
    main()
