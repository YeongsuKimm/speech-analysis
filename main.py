import whisper

model = whisper.load_model("medium")
result = model.transcribe("audio_samples/introduction.mp3")
print(result["text"])
