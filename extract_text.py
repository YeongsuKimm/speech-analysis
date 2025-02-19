import os
import whisper

def transcribe_audio(audio_path):
    """Transcribes the given audio file using Whisper and saves the text."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    text = result["text"]
    
    text_path = os.path.splitext(audio_path)[0] + ".txt"
    with open(text_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    return text_path

def process_audio_files(root_folder: str):
    """Processes all audio files in subfolders, transcribing and saving text."""
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(subdir, file)
                text_path = transcribe_audio(audio_path)
                print(f"Transcribed: {audio_path} -> {text_path}")

# Example usage:
# process_audio_files("/path/to/root/folder")
process_audio_files("data") 