import os
import simpleaudio as sa

class TextAudioPair:
    def __init__(self, text_path: str, audio_path: str):
        """
        Initialize the TextAudioPair with a text file and an audio file.
        :param text_path: Path to the text file.
        :param audio_path: Path to the audio file.
        """
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.text_path = text_path
        self.audio_path = audio_path
    
    def read_text(self) -> str:
        """Reads and returns the content of the text file."""
        with open(self.text_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def play_audio(self):
        """Plays the associated audio file."""
        try:
            wave_obj = sa.WaveObject.from_wave_file(self.audio_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until audio playback is finished
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def __repr__(self):
        return f"TextAudioPair(text='{self.text_path}', audio='{self.audio_path}')"


import os
import json

def find_text_audio_pairs(root_folder: str):
    """Finds and saves pairs of text and audio files with the same name in each subfolder."""
    for subdir, _, files in os.walk(root_folder):
        print(subdir)
        text_audio_pairs = []
        audio_files = {os.path.splitext(f)[0]: os.path.join(subdir, f) for f in files if f.endswith(".mp3")}
        text_files = {os.path.splitext(f)[0]: os.path.join(subdir, f) for f in files if f.endswith(".txt")}
        
        for name in audio_files.keys() & text_files.keys():
            text_audio_pairs.append({"audio": audio_files[name], "text": text_files[name]})
        
        if text_audio_pairs:
            summary_path = os.path.join(subdir, "text_audio_pairs.json")
            with open(summary_path, 'w', encoding='utf-8') as json_file:
                json.dump(text_audio_pairs, json_file, indent=4)
            print(f"Created summary: {summary_path}")

# Example usage:
find_text_audio_pairs("data")
