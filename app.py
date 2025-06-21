import requests
from flask import Flask, request
from flask_talisman import Talisman

app = Flask(__name__)
Talisman(app)

CLIENT_ID = "oav00a2niikmlw46y5v728awzt917i"
CLIENT_SECRET = "wnwvlefbo64r9j877m35xdafeyik5t"

# Get OAuth Token
url = "https://id.twitch.tv/oauth2/token"
params = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "grant_type": "client_credentials"
}
response = requests.post(url, params=params)
access_token = response.json()["access_token"]
print("Access Token:", access_token)

def get_subscribers(broadcaster_id, access_token):
    url = f"https://api.twitch.tv/helix/subscriptions?broadcaster_id={broadcaster_id}"
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        total_subs = data["total"]
        print(f"Active Subscribers: {total_subs}")
        return total_subs
    else:
        print("Error:", response.json())
        return None

# Example usage
BROADCASTER_ID = "ahmpy"  # Get from Twitch API using username
data = get_subscribers(BROADCASTER_ID, access_token)


@app.route("/")
def home():
    return data

if __name__ == "__main__":
    app.run(ssl_context=('cert.pem', 'key.pem'))

