# SPEECH RECOGNITION SYSTEM

A simple yet effective speech-to-text system built using Python. It supports two methods of recognition:
- Google Web Speech API (online)
- Wav2Vec2 (offline) using a pre-trained model from Hugging Face

This allows users to compare cloud-based and local transcription performance on audio inputs.

---

## 🚀 Features

- Accepts WAV audio files as input
- Transcribes using Google’s Web API for quick online recognition
- Uses Wav2Vec2 for offline transcription using deep learning
- Handles different sample rates automatically
- Easy to switch between recognition methods

---

## 🖼️ Screenshot

<img width="1044" alt="Image" src="https://github.com/user-attachments/assets/6bfc7d2e-2555-498c-9ccc-429cf41304f1" />


---

## ⚙️ How it Works

1. **Google Method**:  
   Loads the WAV file using `speech_recognition`, sends it to Google’s servers, and prints the returned transcript.

2. **Wav2Vec2 Method**:  
   Loads the same file, resamples if needed, and passes it through the Wav2Vec2 model for local prediction.

Both methods are implemented in simple Python functions for clarity and reusability.

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/speech-recognition-system.git
cd speech-recognition-system
pip install -r requirements.txt
