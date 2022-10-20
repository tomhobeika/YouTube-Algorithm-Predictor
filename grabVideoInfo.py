from googleapiclient.discovery import build
import requests

api_key = "AIzaSyBN8uq1KrWUHQnmqZOLqamZHcvsy-k9G10" # Don't leak this lol
api_service_name = "youtube"
api_version = "v3"
youtube = build(api_service_name, api_version, developerKey=api_key)

request = youtube.videos().list(
	part="statistics,snippet",
	id="oneyh3bDmw4"
)
video = request.execute()

views = video["items"][0]["statistics"]["viewCount"]
channelId = video["items"][0]["snippet"]["channelId"]
thumb = video["items"][0]["snippet"]["thumbnails"]["standard"]["url"]

print(views)
print(channelId)
print(thumb)

request = youtube.channels().list(
	part="statistics",
	id=channelId
)
channel = request.execute()

subs = channel["items"][0]["statistics"]["subscriberCount"]
print(subs)

thumb_data = requests.get(thumb).content
with open(views + '_' + subs + '.jpg', 'wb') as handler:
	handler.write(thumb_data)