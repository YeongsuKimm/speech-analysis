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
import sys

def mkv_to_wav(input_file, output_file):
    try:
        ffmpeg.input(input_file, ss=0, t=1800).output(output_file, acodec='pcm_s16le', ar=44100, ac=2).run()
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error occurred: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def download_twitch_video(url, streamer_name):
    command = f"twitch-dl download {url}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
    
    # Read the initial output to capture the options
    stdout, stderr = process.communicate(timeout=5)

    # Look for the available options for video quality/resolution
    options = []
    for line in stdout.splitlines():
        match = re.match(r"(\d+)\) (.+)", line)
        if match:
            key = match.group(1)
            option_name = match.group(2)
            options.append((key, option_name))

    # Find the key for 'Audio Only'
    audio_only_key = None
    for key, option_name in options:
        if 'Audio Only' in option_name:
            audio_only_key = key
            break

    if audio_only_key:
        print(f"Selecting option: {audio_only_key} for 'Audio Only'")

        # Start the process again to allow writing input and reading output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)

        # Send the audio-only option key
        process.stdin.write(f"{audio_only_key}\n")
        process.stdin.flush()

        # Wait for the process to finish and get the output
        stdout, stderr = process.communicate()

        print("All tasks completed, running final command...")
        # subprocess.run(["echo", "Final cleanup command"])

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
    else:
        print("No 'Audio Only' option found.")
    
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

def extract_main_speaker_audio(input_audio, speaker_timestamps, main_speaker, streamer_name, strt, min_duration=2, max_duration=20, merge_gap=2):
    output_folder = f"data/{streamer_name}"
    os.makedirs(output_folder, exist_ok=True)
    audio = AudioSegment.from_wav(input_audio)

    # Sort timestamps
    timestamps = sorted(speaker_timestamps[main_speaker], key=lambda x: x[0])

    merged_segments = []
    current_start, current_end = timestamps[0]

    for i in range(1, len(timestamps)):
        start, end = timestamps[i]

        # Check if the gap is small enough to merge
        if start - current_end <= merge_gap:
            current_end = end  # Extend the segment
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end

    # Add last segment
    merged_segments.append((current_start, current_end))

    # Export merged segments, ensuring each is within min/max duration
    clip_idx = 0
    for start, end in merged_segments:
        duration = end - start
        if duration < min_duration:
            continue  # Skip very short clips

        if duration > max_duration:
            end = start + max_duration  # Trim to max duration

        start_ms, end_ms = int(start * 1000), int(end * 1000)
        clip = audio[start_ms:end_ms]
        clip.export(f"{output_folder}/{strt}clip_{clip_idx}.mp3", format="mp3")
        clip_idx += 1

    print(f"Extracted {clip_idx} clips (each {min_duration}-{max_duration}s) for {streamer_name}.")


def process_twitch_audio(audio_path, streamer_name, strt):
    speaker_timestamps = diarize_audio(audio_path)
    main_speaker = identify_main_speaker(speaker_timestamps)
    extract_main_speaker_audio(audio_path, speaker_timestamps, main_speaker, streamer_name, strt)

def main(url, streamer_name,strt):
    audio_path = download_twitch_video(url, streamer_name)
    if audio_path:
        process_twitch_audio(audio_path, streamer_name, strt)
    os.remove(f"data/{streamer_name}/output.wav")

completed=[]
with open("completed.txt", "r") as file:
    for line in file:
        if "\n" in line:
            completed.append(str(line)[0:-1])
        else:
            completed.append(str(line))
print(completed)

failed=[]
with open("failed.txt", "r") as file:
    for line in file:
        if "\n" in line:
            failed.append(str(line)[0:-1])
        else:
            failed.append(str(line))
print(failed)

streamers = []
with open("test.txt", "r") as file:
        for streamer in file:
            streamers.append(streamer[:-1])

# Divide into 2 equal batches
batch_size = len(streamers) // 2

batch1 = streamers[:batch_size]
batch2 = streamers[batch_size:]

# Print results
print("Batch 1:", batch1)
print("Batch 2:", batch2)

for name in batch2:
    i=1
    if name not in completed:
        print(name)
        try:
            with open(f"vods/{name}.txt", "r") as file:
                for line in file:
                    try:
                        main(line, name,i)
                    except BrokenPipeError and FileNotFoundError and ValueError:
                        with open("completed.txt","a") as file2:
                            file2.write("\n"+name)
                        with open("failed.txt","a") as file2:
                            file2.write("\n"+name)
                        # rerun this file
                        print(f"Error encountered. Restarting script in 5 seconds...")
                        time.sleep(5)  # Optional delay before restart

                        subprocess.run([sys.executable, "auto_data.py"])
                        break
                    i+=1;
                with open("completed.txt","a") as file2:
                    file2.write("\n"+name)
        except:
            with open("completed.txt","a") as file2:
                file2.write("\n"+name)
            with open("failed.txt","a") as file2:
                file2.write("\n"+name)
            # rerun this file
            print(f"Error encountered. Restarting script in 5 seconds...")
            time.sleep(5)  # Optional delay before restart

            subprocess.run([sys.executable, "auto_data.py"])

from extract_text import process_audio_files
# from TextAudioPair import find_text_audio_pairs

process_audio_files("data") 
# find_text_audio_pairs("data")

