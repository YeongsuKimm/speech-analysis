import os
import subprocess
import time
import ffmpeg
import torch
import re
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from pyannote.audio import Model
from pydub import AudioSegment

def mkv_to_wav(input_file, output_file):
    try:
        ffmpeg.input(input_file, ss=0, t=900).output(output_file, acodec='pcm_s16le', ar=44100, ac=2).run()
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error: {e}")
        print(e.stderr.decode())

def download_twitch_video(url, streamer_name):
    command = f"twitch-dl download {url}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
    time.sleep(3)
    process.stdin.write('6\n')
    process.stdin.flush()
    stdout, stderr = process.communicate()
    print("All tasks completed, running final command...")
    subprocess.run(["echo", "Final cleanup command"])

    if stdout:
        print("Download output:", stdout)
        match = re.search(r"Downloaded: (\S+)", stdout)
        if match:
            downloaded_file = match.group(1)
            print("Downloaded file:", downloaded_file)
            output_wav = f"data/{streamer_name}/output.wav"
            mkv_to_wav(downloaded_file, output_wav)
            os.remove(downloaded_file)
            return output_wav
        else:
            print("File name not found in the output.")
    if stderr:
        print("Error Output:", stderr)
    return None

pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
pipeline.to(torch.device("cuda"))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)
    speaker_timestamps = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_timestamps:
            speaker_timestamps[speaker] = []
        speaker_timestamps[speaker].append((turn.start, turn.end))
    return speaker_timestamps

def identify_main_speaker(speaker_timestamps):
    speaker_durations = {speaker: sum(end - start for start, end in times) for speaker, times in speaker_timestamps.items()}
    main_speaker = max(speaker_durations, key=speaker_durations.get)
    return main_speaker

def extract_main_speaker_audio(input_audio, speaker_timestamps, main_speaker, streamer_name, strt):
    output_folder = f"data/{streamer_name}"
    os.makedirs(output_folder, exist_ok=True)
    audio = AudioSegment.from_wav(input_audio)
    for idx, (start, end) in enumerate(speaker_timestamps[main_speaker]):
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        clip = audio[start_ms:end_ms]
        clip.export(f"{output_folder}/{strt}clip_{idx}.mp3", format="mp3")
    print(f"Extracted {len(speaker_timestamps[main_speaker])} clips of the main speaker for {streamer_name}.")

def process_twitch_audio(audio_path, streamer_name, strt):
    speaker_timestamps = diarize_audio(audio_path)
    main_speaker = identify_main_speaker(speaker_timestamps)
    extract_main_speaker_audio(audio_path, speaker_timestamps, main_speaker, streamer_name, strt)

def main(url, streamer_name,strt):
    audio_path = download_twitch_video(url, streamer_name)
    if audio_path:
        process_twitch_audio(audio_path, streamer_name, strt)
    os.remove(f"data/{streamer_name}/output.wav")

completed=["ahmpy","aircool","AuzioMF","bateson87","Beardageddon","BennyCentral","BikeMan","Blue_Squadron","BreaK","BreesKnees","BrownGotti","Caedrel","carmen","caseoh_","CDawgVA","cjya","Couriway","crazyjapanese","d0cc_tv","DEFAC3D","Elajjaz","erobb221","Eros","Everretta",
           "Fannsy","Geef","Gnomonkey","Gorgc","HasanAbi","HollywoodBob","huncho","iddqd","ixxdeee","J4CKIECHAN","Jacque","JayOddity","JonSandman","Jynxzi","k3soju","KaiCenat","Kerrty","KmartPoker","kyliebitkin","Lacy"]
# LACY IS NOT COMPLETE FIGURE OUT WHAT IS WRONG I THINK THE STREAM IS OUTDATED
for name in os.listdir("vods/"):
    i=1
    if name not in completed:
        with open(f"vods/{name}", "r") as file:
            for line in file:
                print(name);
                main(line, name,i)
                i+=1;

from extract_text import process_audio_files
from TextAudioPair import find_text_audio_pairs
process_audio_files("data")
find_text_audio_pairs("data")