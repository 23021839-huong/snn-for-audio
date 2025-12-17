\# SNN for Audio Classification



This project implements a Spiking Neural Network (SNN) for audio processing and classification.



\## Project Pipeline

1\. Convert audio to WAV format

2\. Preprocess audio (feature extraction)

3\. Train SNN model

4\. Predict on test audio



\## Source Code

\- `convert\_to\_wav.py` : Convert raw audio to wav

\- `Preprocess\_audio.py` : Audio preprocessing

\- `Train\_snn.py` : Train SNN model

\- `predict.py` : Run inference on audio



\## Requirements

\- Python 3.8+

\- PyTorch

\- snnTorch

\- librosa

\- numpy



\## Run

```bash

python Train\_snn.py

python predict.py



