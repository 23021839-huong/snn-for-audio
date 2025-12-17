import os
from pydub import AudioSegment

root = r"E:\lab\SnnforAudio\data_voice"


for label in os.listdir(root):
    folder = os.path.join(root, label)

    for file in os.listdir(folder):
        if file.endswith(".m4a"):
            path = os.path.join(folder, file)
            wav_path = path.replace(".m4a", ".wav")

            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio.export(wav_path, format="wav")

            print("Converted:", wav_path)
