# import os
# for name in os.listdir("data/"):
#     print(name)
#     if name == "audio_conv.py":
#         continue;
#     with open(f"data/{name}/{name}.txt", "r") as file:
#         current = file.readline()[21:]
#         alltime = file.readline()[34:]
#         try:
#             current = int(current)
#         except:
#             with open(f"scrap.txt", "a") as file2:
#                 file2.write(name+"\n")
#                 continue
#         try: 
#             alltime = int(alltime)
#         except:
#             with open(f"scrap.txt", "a") as file2:
#                 file2.write(name+"\n")
#                 continue




# with open("scrap.txt", "r") as file:
#     scrap_names = [line.strip() for line in file]
#     for name in os.listdir("processed/"):
#         if name not in scrap_names:
#             continue;
#         else:
#             shutil.rmtree(f"processed/{name}")

# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime, timedelta
# import os
# import subprocess
# import re

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# }

# streamer_names = []
# invalid_names = ["Riot Games", "ESLCS", "Fortnite", "VALORANT", "RocketLeague", "easportsfc", "TwitchRivals", "Twitch", 
#                  "Warframe", "GenshinImpactOfficial", "Call of Duty", "BLASTPremier", "LEC", "ow_esports", "GamesDoneQuick", 
#                  "LCK", "PGL", "ESLCS_GG", "FACEIT TV", "ELEAGUE TV", "thegameawards", "NASA", "BotezLive", "ESLCSb", "dota2ti", 
#                  "Chess", "BeyondTheSummit", "PUBG_BATTLEGROUNDS", "BibleBoysChurch"]
# completed = []
# with open("completed.txt", "r") as file:
#     for name in file:
#         completed.append(name[:-1])

# # print(completed)

# # Ensure the directory exists
# os.makedirs("data", exist_ok=True)


# def check_recent_streams(streamer_name):
#     # Run the twitch-dl command and capture the output
#     result = subprocess.run(["twitch-dl", "videos", streamer_name], capture_output=True, text=True)
    
#     # Regular expression to capture video info (video ID, length, and link)
#     pattern = re.compile(r"Video (\d+).*?Length: (\d+) h (\d+) min.*?(https://www.twitch.tv/videos/\d+)", re.DOTALL)
    
#     # Regular expression to capture the publication date and time
#     date_pattern = r"Published (\d{4}-\d{2}-\d{2}) @ (\d{2}:\d{2}:\d{2})"
    
#     recent_count = 0
#     current_date = datetime.today()
#     max_days_ago = timedelta(days=40)

#     for match in pattern.finditer(result.stdout):
#         # Get publication date and time for each match
#         for m in re.findall(date_pattern, result.stdout):
#             publication_date_str, publication_time = m
#             publication_date = datetime.strptime(publication_date_str, "%Y-%m-%d")
            
#             # Calculate the difference between current date and publication date
#             date_difference = current_date - publication_date
            
#             # If the video was published in the last 40 days
#             if date_difference <= max_days_ago:
#                 video_id, hours, minutes, link = match.groups()
#                 hours, minutes = int(hours), int(minutes)

#                 # Filter videos longer than 2 hours
#                 if hours > 2 or (hours == 2 and minutes > 0):
#                     recent_count += 1
#                     if recent_count == 3:
#                         break
    
#     return recent_count >= 3


# for tab in range(1, 6):    
#     suburl = f"https://twitchtracker.com/channels/most-followers/english?page={tab}"
#     response = requests.get(suburl, headers=headers)
    
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # Find all table rows containing streamer data
#         rows = soup.find_all('tr')
        
#         for row in rows:
#             name_td = row.find_all('td')
#             if len(name_td) >= 3:  # Ensuring the correct column exists
#                 name_tag = name_td[2].find('a')  # The streamer name is in the 3rd <td> (index 2)
#                 if name_tag:
#                     streamer_name = name_tag.text.strip()
#                     if streamer_name and streamer_name not in streamer_names:
#                         if streamer_name not in completed and streamer_name not in invalid_names:
#                             if check_recent_streams(streamer_name):
#                                 print(streamer_name)
#                                 streamer_names.append(streamer_name)
#                                 with open("todo.txt", "a", encoding="utf-8") as file:
#                                     file.write(streamer_name + "\n")
#     else:
#         print(f"Failed to fetch page {tab}")

# print(f"Total active streamers scraped: {len(streamer_names)}")



# import os

# # Read scraped names from scrap.txt and store as a set
# with open("scrap.txt", "r") as file:
#     scrap_names = {line.strip() for line in file}  # Convert to set

# # Get list of existing streamers and store as a set
# existing_streamers = set(os.listdir("data/"))

# # Find missing streamers (those in scrap.txt but not in data/)
# missing_streamers = scrap_names.difference(existing_streamers)  # Set difference

# # Write missing names to todo.txt
# if missing_streamers:
#     with open("todo.txt", "w") as todo_file:
#         todo_file.write("\n".join(missing_streamers) + "\n")

# print(f"Added {len(missing_streamers)} new streamers to todo.txt.")


# import os

# def clean_vods_folder(data_folder="data", vods_folder="vods"):
#     # Get the list of subfolder names in 'data'
#     if not os.path.exists(data_folder):
#         print(f"The folder '{data_folder}' does not exist.")
#         return
    
#     data_subfolders = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
#     print(data_subfolders)
#     # Check if 'vods' folder exists
#     if not os.path.exists(vods_folder):
#         print(f"The folder '{vods_folder}' does not exist.")
#         return
    
#     # Iterate through files in 'vods' folder
#     for file in os.listdir(vods_folder):
#         # print(file)
#         file_path = os.path.join(vods_folder, file)
#         print(file_path)
#         # Ensure it's a text file
#         if file not in data_subfolders:
#             os.remove(file_path)

# if __name__ == "__main__":
#     clean_vods_folder()



# if __name__ == "__main__":
    # get_vods("Casson")


# import os
# import subprocess
# import re

# def make_vods():
#     with open("todo.txt", "r") as file:
#         for streamer in file:
#             open(f"vods/{streamer.strip()}.txt", "w")

# make_vods()

# streamers = []
# with open("todo.txt", "r") as file:
#         for streamer in file:
#             streamers.append(streamer[:-1])

# def get_vods(streamer_name):
#     # Run the twitch-dl command and capture the output
#     result = subprocess.run(["twitch-dl", "videos", streamer_name], capture_output=True, text=True)
#     pattern = re.compile(r"Video (\d+).*?Length: (\d+) h (\d+) min.*?(https://www.twitch.tv/videos/\d+)", re.DOTALL)

#     # Extract and filter videos
#     i = 0;
#     list = []
#     for match in pattern.finditer(result.stdout):
#         if i == 3:
#             break
#         video_id, hours, minutes, link = match.groups()
#         hours, minutes = int(hours), int(minutes)

#         # Filter videos longer than 2 hours
#         if hours > 2 or (hours == 2 and minutes > 0):
#             # print(f"{video_id}: {link}")
#             i+=1
#             list.append(link)
#     return list

# delete = []

# with open("test.txt", "w") as file:
#     for streamer in streamers:
#         print(streamer)
#         vods = get_vods(streamer)
#         print(vods)
#         if len(vods) >= 3:
#             file.write(streamer+"\n")
# file.close()

# streamers = []
# with open("test.txt", "r") as file:
#         for streamer in file:
#             streamers.append(streamer[:-1])

# def store_vod_links():
#     for streamer in streamers:
#         with open(f"vods/{streamer}.txt", "r+") as file:
#             # if file.readline() == "":
#             print(streamer)
#             vods = get_vods(streamer)
#             print(vods)
#             for vod_link in vods:
#                 file.write(vod_link+"\n")


# store_vod_links()

# print(delete)
# streamers = []


# streamers = []
# with open("test.txt", "r") as file:
#         for streamer in file:
#             streamers.append(streamer[:-1])


# with open('completed.txt', "r") as file:
#     for name in file:
#         streamers.append(name[:-1])
# file.close()

# delete = []
# with open("todo.txt", "r") as file:
#     for name in file:
#         if name[:-1] not in streamers:
#             delete.append(name[:-1])
# file.close()

# print(delete)


# streamers = [
#     "hypnoshark", "BarbarousKing", "Northernlion", "Casson", "ExtraEmily", "ImperialHal__",
#     "MataraKan", "DGthe99", "TobiasFate", "CohhCarnage", "Keeoh", "kyootbot", "Emiru",
#     "ironmouse", "Nmplol", "fl0m", "BruceGreene", "NICKMERCS", "MOONMOON", "L3WG",
#     "Lord_Kebun", "Fanum", "Shotz", "iLumpE", "yourragegaming", "RDCgaming", "Lacy",
#     "angryginge13", "AxialMatt", "Glorious_E", "Maximum", "Insym", "jordansisco_", "T90Official",
#     "robcdee", "GernaderJake", "Philza", "ThePrimeagen", "filian", "scump", "LIRIK",
#     "Zoomaa", "Sterdekie", "Hungrybox", "omareloff", "lilsimsie", "Ludwig",
#     "PontiacMadeDDG", "supertf", "KingWoolz", "Camy", "Tubbo", "Emongg", "Maximilian_DOOD",
#     "AmericanDad", "InternetCityArcade", "AussieAntics", "itsSpoit", "jasontheween",
#     "Bigpuffer", "Ray__C", "Aztecross", "vedal987", "Adapt", "Clix",
#     "BobbyRayGray", "Ray", "MurderCrumpet", "plaqueboymax", "capturesca", "peterpark",
#     "MeatyMarley", "JuicyJohns", "LVNDMARK", "Mactics", "Joe_Bartolozzi", "KidShadoe",
#     "PaymoneyWubby", "BobbyPoffGaming", "Rain"
# ]


# import os
# for streamer in streamers:
#     os.makedirs(f"data/{streamer}", exist_ok=True)


# streamers = []
# import os
# for streamer in os.listdir("processed/"):
#     streamers.append(streamer)

# print(len(streamers))

# streamers = []
# with open("completed.txt", "r") as file:
#     for name in file:
#         if "\n" in name:
#             streamers.append(name[:-2])

# print(streamers)


# streamers = []
# audio = []
# text = []
# import os
# for streamer in os.listdir("processed/"):
#     streamers.append(streamer)
#     atotal = 0
#     ttotal = 0
#     for file in os.listdir(f"processed/{streamer}"):
#         if file.startswith("audio"):
#             atotal += 1
#         if file.startswith("text"):
#             ttotal += 1
#     audio.append(atotal)
#     text.append(ttotal)
#     if atotal < 50:
#         print(streamer + " : " +str(atotal))
    
# # print(audio)
# # print(text)
# # print(audio==text)
# print(audio)
# print(len(audio))

# streamers = []
# audio1 = []
# import os
# for streamer in os.listdir("data/.test/"):
#     if streamer == "audio_conv.py":
#         continue
#     streamers.append(streamer)
#     atotal = 0
#     # print(streamer)
#     for file in os.listdir(f"data/.test/{streamer}"):
#         if file.endswith("mp3"):
#             atotal += 1
#     audio1.append(atotal)
#     if atotal < 50:
#         print(streamer + " : " +str(atotal))

# print(audio1)
# print(len(audio1))

# import os

# with open("completed.txt", "w") as f:
#     for streamer in os.listdir("data/"):
#         f.write(streamer+"\n")


# import os
# with open("test.txt", "r") as file:
#     for streamer in file:
#         os.mkdir(f"data/.test/{streamer[:-1]}")

# import os

# for streamer in os.listdir("data/.test/"):
#     with open("test.txt", "a") as file:
#         file.write(streamer + "\n")

# import torch
# data = torch.load(".pt")  # Replace with actual file path
# print(type(data))
# Debug sample loading


# import random
# num = []
# for i in range(0,19):
#     number = random.randint(1,183)
#     while number in num:
#         number = random.randint(1,183)
#     num.append(number)

# print(len(num))

# import os
# streamers = []
# for streamer in os.listdir("processed_copy"):
#     streamers.append(streamer)

# print(len(streamers))

# print(num)

# selected = []
# for i in num:
#     selected.append(streamers[i-1])

# print(selected)


# import shutil
# for i in selected:
#     source_folder = f'processed_copy/{i}'
#     destination_folder = 'shadowrealm'
#     try:
#         shutil.move(source_folder, destination_folder)
#     except:
#         continue


# import shutil
# import os
# from classification import get_dict
# streamer_list = os.listdir("processed")

# # streamer_list.append(os.listdir(".test"))
# streamer_list2 = os.listdir(".test")
# # streamer_list.extend(streamer_list2)
# label_dict = get_dict(streamer_list)
# # print(label_dict)
# one = []
# two = []
# three = []
# for i in label_dict:
#     if label_dict[i]==0:
#         one.append(i)
#     elif label_dict[i]==1:
#         two.append(i)
#     elif label_dict[i]==2:
#         three.append(i)
#     else:
#         print(i)
#         print(label_dict[i])
# print(len(one))
# print(len(two))
# print(len(three))
# print()



def distribute_three_lists_into_five_parts(list1, list2, list3):
    all_items = [
        ("list1", item) for item in list1
    ] + [
        ("list2", item) for item in list2
    ] + [
        ("list3", item) for item in list3
    ]

    total = len(all_items)  # 244
    base_chunk = total // 5
    remainder = total % 5  # some groups will get 1 extra

    parts = [[] for _ in range(5)]
    idx = 0
    for i in range(5):
        size = base_chunk + (1 if i < remainder else 0)
        for _ in range(size):
            parts[i].append(all_items[idx][1])
            idx += 1

    return parts


# parts = distribute_three_lists_into_five_parts(one,two,three)
# # print(len(parts))

# fold_1 = parts[0]
# fold_2 = parts[1]
# fold_3 = parts[2]
# fold_4 = parts[3]
# test = parts[4]

# print(fold_1)
# print(fold_2)
# print(fold_3)
# print(fold_4)
# print(test)

# print(distribution(fold_1))

# test = []

# Count all folders in "processed"


# for name in test_folders:
#     source_folder = os.path.join(".test", name)
#     destination_folder = os.path.join("processed", name)

#     if not os.path.exists(source_folder):
#         print(f"❌ {name} not found in processed/")
#         continue

#     try:
#         shutil.move(source_folder, destination_folder)
#         print(f"✅ Moved {name} to .test/")
#     except Exception as e:
#         print(f"⚠️ Error moving {name}: {e}")

# import os
# import glob

# root_dir = 'processed'  # your main folder

# # This will find all .pt files inside all subfolders of 'processed'
# for file_path in glob.iglob(os.path.join(root_dir, '*', '*.pt')):
#     try:
#         os.remove(file_path)
#         print(f"Deleted: {file_path}")
#     except Exception as e:
#         print(f"Failed to delete {file_path}: {e}")



# mset = set()
# with open("scrap.txt","r") as f:
#     for line in f:
#         start = line.find("/")
#         end = line.find("\\")
#         mset.add(line[start+1:end])
        
# print(mset)

import os
# streamers = os.listdir("processed")
# print(streamers)
# print(len(streamers))
# for streamer in os.listdir("processed"):
#     for name in os.listdir(f"processed/{streamer}"):
#         if name.endswith(".pt"):
#             file_path = os.path.join("processed", streamer, name)
#             # print(name)
#             os.remove(file_path)
#             print(f"Deleted: {file_path}")

# processed_folders = os.listdir("processed")
# print(f"Total folders in 'processed/': {len(processed_folders)}")
# import shutil
# import os
# test_folders = os.listdir(".test")
# # print(test_folders)
# print(f"Total folders in '.test/': {len(test_folders)}")
# # Move test streamers to .test folder
# for name in test_folders:
#     source_folder = os.path.join(".test", name)
#     destination_folder = os.path.join("processed", name)

#     if not os.path.exists(source_folder):
#         print(f"❌ {name} not found in processed/")
#         continue

#     try:
#         shutil.move(source_folder, destination_folder)
#         print(f"✅ Moved {name} to .test/")
#     except Exception as e:
#         print(f"⚠️ Error moving {name}: {e}")



import os
from classification import get_dict

combined = os.listdir('processed')
fold_1 = []
fold_2 = []
fold_3 = []
fold_4 = []
test = []

def distribution(streamer_list):
    label_dict = get_dict(combined)
    one = []
    two = []
    three = []
    for i in streamer_list:
        if label_dict[i]==0:
            one.append(i)
        elif label_dict[i]==1:
            two.append(i)
        elif label_dict[i]==2:
            three.append(i)
        else:
            print(i)
            print(label_dict[i])
    print(len(one))
    print(len(two))
    print(len(three))

label_dict = get_dict(combined)
one = []
two = []
three = []
for i in label_dict:
    if label_dict[i]==0:
        one.append(i)
    elif label_dict[i]==1:
        two.append(i)
    elif label_dict[i]==2:
        three.append(i)
    else:
        print(i)
        print(label_dict[i])

while(len(one) >= 5):
    fold_1.append(one.pop())
    fold_2.append(one.pop())
    fold_3.append(one.pop())
    fold_4.append(one.pop())
    test.append(one.pop())
    
    fold_1.append(two.pop())
    fold_2.append(two.pop())
    fold_3.append(two.pop())
    fold_4.append(two.pop())
    test.append(two.pop())
    
    fold_1.append(three.pop())
    fold_2.append(three.pop())
    fold_3.append(three.pop())
    fold_4.append(three.pop())
    test.append(three.pop())


fold_1.append(one.pop())
fold_2.append(two.pop())
fold_3.append(two.pop())
fold_4.append(three.pop())

print(len(fold_1))
print(len(fold_2))
print(len(fold_3))
print(len(fold_4))
print(len(test))    

one = 0
two = 0
three = 0
label_dict = get_dict(combined)

print(fold_1)
print(fold_2)
print(fold_3)
print(fold_4)
print(test)

# for streamer in combined:
#     print(streamer +": " + str(len(os.listdir(f"data/{streamer}"))) + " (" +str(label_dict[streamer]) + ")")
#     if label_dict[streamer] == 0:
#         one += len(os.listdir(f"data/{streamer}"))
#     if label_dict[streamer] == 1:
#         two += len(os.listdir(f"data/{streamer}"))
#     if label_dict[streamer] == 2:
#         three += len(os.listdir(f"data/{streamer}"))
#     # if len(os.listdir(f"processed/{streamer}")) == 1:
#     #     with open("todo.txt", "a") as file:
#     #         file.write(streamer+"\n")
# print(one)
# print(two)
# print(three)
# print(len(combined))
