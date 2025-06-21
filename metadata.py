import os
import requests
from bs4 import BeautifulSoup
import re
import time

nosub = []

def get_subs_and_followers(name):
    time.sleep(1)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    suburl = f"https://twitchtracker.com/{name}"
    response = requests.get(suburl, headers=headers)
    print(suburl)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        current_active_subs = False
        all_time_high_active_subs = False
        total_followers = False
        try:
            blocks = soup.find_all('div', class_='g-x-s-block')
            for block in blocks:
                current_active_subs_label = block.find('div', class_='g-x-s-label')
                current_active_subs = block.find('div', class_='g-x-s-value')
                if current_active_subs_label and current_active_subs:
                    if re.search(r'Current active subs', current_active_subs_label.text.strip(), re.IGNORECASE):
                        print(f"Current Active Subs: {current_active_subs.text.strip()}")
                else:
                    print("Current active subs not found.")
        
            # All-time High Active Subs
            all_time_high_active_subs_label = soup.find('div', class_='g-x-s-label', string=re.compile(r'.*All-time high active subs.*', re.IGNORECASE))
            if all_time_high_active_subs_label:
                all_time_high_active_subs = all_time_high_active_subs_label.find_previous('div', class_='g-x-s-value')
                if all_time_high_active_subs:
                    print(f"All-time High Active Subs: {all_time_high_active_subs.text.strip()}")
                else:
                    print("All-time High Active Subs value not found.")
            else:
                print("All-time high active subs label not found.")

            # Total Followers
            total_followers_label = soup.find('div', class_='g-x-s-label', string=re.compile(r'.*Total followers.*', re.IGNORECASE))
            if total_followers_label:
                total_followers = total_followers_label.find_previous('div', class_='g-x-s-value').find('span', class_='to-number')
                if total_followers:
                    print(f"Total Followers: {total_followers.text.strip()}")
                else:
                    print("Total Followers value not found.")
            else:
                print("Total followers label not found.")
            
            # Save data to a text file
            with open(f"data/{name}/{name}.txt", 'w') as file:
                file.write(f"Current Subscribers: {current_active_subs.text.strip() if current_active_subs else 'N/A'}\n")
                file.write(f"All-Time High Active Subscribers: {all_time_high_active_subs.text.strip() if all_time_high_active_subs else 'N/A'}\n")
                file.write(f"Total Followers: {total_followers.text.strip() if total_followers else 'N/A'}\n")
                
        except Exception as e:
            print(f"Error occurred: {e}")
            return
    else:
        print(f"Request failed with status code {response.status_code}")

streamers = []
with open("completed.txt", "r") as file:
        for streamer in file:
            streamers.append(streamer[:-1])

# streamers = ["broxh_","Chap", "Duke", "LTANorth", "Nadia", "Rainbow6", "Valkyrae"]

for name in streamers:
    get_subs_and_followers(name)

