import ffmpeg
import os

input_folder = "zzz.videos"
output_folder = "miia"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a .txt file with the same name as the output folder
txt_file_path = os.path.join(output_folder, f"{output_folder}.txt")

# Write the information into the .txt file
with open(txt_file_path, "w") as txt_file:
    txt_file.write("Hours streamed: \n")
    txt_file.write("Average viewers: \n")
    txt_file.write("Peak viewers: \n")
    txt_file.write("Hours watched: \n")
    txt_file.write("Followers gained: \n")
    txt_file.write("Followers / hour: \n")
    print(f"Created {txt_file_path} with placeholders.")

# Process the video files and convert them to audio
for file in os.listdir(input_folder):
    if file.endswith(".mp4"):  # Change as needed
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".mp3")

        try:
            # Convert video to audio
            ffmpeg.input(input_path).output(output_path, format='mp3', acodec='libmp3lame').run()
            print(f"Converted: {file} to MP3")
            
            # Delete the original video file
            os.remove(input_path)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
