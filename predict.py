import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# 1. Energy-based Voice Activity Detection
def vad_energy(audio, sr, frame_ms=30, threshold=0.0015):
    frame_len = int(sr * frame_ms / 1000)
    energies = []

    for i in range(0, len(audio), frame_len):
        frame = audio[i:i + frame_len]
        energies.append(np.sum(frame ** 2))

    energies = np.array(energies)
    speech_idx = np.where(energies > threshold)[0]

    if len(speech_idx) == 0:
        return audio

    start = speech_idx[0] * frame_len
    end = (speech_idx[-1] + 1) * frame_len
    return audio[start:end]


# 2. Record audio from microphone
def record_to_file(out_path="record_test.wav", duration=1.5, sr=16000):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    trimmed = vad_energy(audio, sr)

    if len(trimmed) < 3000:
        trimmed = audio

    sf.write(out_path, trimmed, sr)
    return out_path


# 3. MFCC preprocessing
def preprocess_file(path, max_len=50, n_mfcc=40):
    audio, sr = librosa.load(path, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-9)

    if mfcc.shape[1] < max_len:
        pad = np.zeros((n_mfcc, max_len - mfcc.shape[1]))
        mfcc = np.concatenate([mfcc, pad], axis=1)
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc.T  # (Time, Features)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)


# 4. Audio SNN model (1–9)
class AudioSNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(40, 128)
        self.lif1 = snn.Leaky(beta=0.9,
                              spike_grad=surrogate.fast_sigmoid(slope=25))

        self.fc2 = nn.Linear(128, 9)
        self.lif2 = snn.Leaky(beta=0.9,
                              spike_grad=surrogate.fast_sigmoid(slope=25))

    def forward(self, x):
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

        return torch.stack(spk_out, dim=0)


# 5. Prediction
def predict(model, mfcc):
    with torch.no_grad():
        spk_rec = model(mfcc)
        out = spk_rec.sum(dim=0)

        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        return pred + 1, probs.cpu().numpy()


# 6. Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--duration", type=float, default=1.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioSNN().to(device)
    model.load_state_dict(torch.load("snn_audio_model.pt", map_location=device))
    model.eval()

    if args.record:
        input_path = record_to_file(duration=args.duration)
    elif args.file:
        input_path = args.file
    else:
        print("Please use --record or --file <path>")
        return

    mfcc = preprocess_file(input_path).to(device)
    pred, probs = predict(model, mfcc)

    print("File:", input_path)
    print("Predicted digit:", pred)
    print("Softmax probabilities:", probs)


if __name__ == "__main__":
    main()
