import os
import subprocess

folder = r"E:\lab\SnnforAudio\data_voice\5"

for file in os.listdir(folder):
    if file.endswith(".wav"):
        subprocess.run([
            "python", "predict.py",
            "--file", os.path.join(folder, file)
        ])
