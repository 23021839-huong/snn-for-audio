import os
import shutil

SRC = r"E:\free-spoken-digit-dataset-master\recordings"
DST = r"E:\lab\SnnforAudio\data_voice"

# tạo folder 1 → 9 nếu chưa có
for i in range(1, 10):
    os.makedirs(os.path.join(DST, str(i)), exist_ok=True)

count = 0

for file in os.listdir(SRC):
    if file.endswith(".wav"):
        digit = file.split("_")[0]  # lấy số đầu file
        if digit in [str(i) for i in range(1, 10)]:
            shutil.copy(
                os.path.join(SRC, file),
                os.path.join(DST, digit, file)
            )
            count += 1

print(f"Đã copy xong {count} file WAV vào data_voice/")
