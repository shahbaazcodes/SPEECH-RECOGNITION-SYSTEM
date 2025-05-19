import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import soundfile as sf

# Method 1: Using SpeechRecognition and Google Web API
def recognize_with_google(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Google Speech Recognition:")
        print(text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google; {e}")

# Method 2: Using Wav2Vec2 from HuggingFace
def recognize_with_wav2vec(audio_file):
    print("\nWav2Vec2 Recognition:")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    speech, rate = sf.read(audio_file)
    if rate != 16000:
        raise ValueError("Audio must be 16kHz.")

    input_values = tokenizer(speech, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    print(transcription.lower())

if __name__ == "__main__":
    audio_path = "audio_samples/sample.wav"
    recognize_with_google(audio_path)
    recognize_with_wav2vec(audio_path)
