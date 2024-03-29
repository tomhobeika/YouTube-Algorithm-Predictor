from googleapiclient.discovery import build
import requests
import os

api_key = "AIzaSyBN8uq1KrWUHQnmqZOLqamZHcvsy-k9G10" # Don't leak this lol

youtube = build("youtube", "v3", developerKey=api_key)

# Takes numPerPlaylist videos for each videoID
numPerPlaylist = 100
#playlists = {"UUFrj6EBhnHlWqmvuwH1Ue4g","UUIKlsX1qfGqKPt4KAW-JOZg","UUGMqfwchF-3lLX443rY5Ygg", "UUeuPD9Hx5og0KNy7ZTq9Wew", "UUR1D15p_vdP3HkrH8wgjQRw", "UUX6OQ3DkcsbYNE6H8uQQuVA", "UUp6F1mQGuaXiDj5otWBmIjg", "UUqTYHSnBUXZamsVcOlQf-fg",
#	"UUPcFg7aBbaVzXoIKSNqwaww", "UUcnci5vbpLJ-rh_V9yXwawg"}

playlists = {
	"UUtMVHI3AJD4Qk4hcbZnI9ZQ", "UUkDSAQ_5-yx5hmuvUcsJL7A", "UUzH3sYjz7qi6o1HFPRD0HCQ",
	"UU00uG71I6iPyx15EX6i_GDA", "UUJHA_jMfCvEnv-3kRjTCQXw", "PLTvFd80YJldFsrP6QQzen2JXnM7dy0nLa",
	"UUeshNiRSXKg2Fk7BeREgR2g", "UUXuqSBlHAE6Xw-yeJA0Tunw", "UUJFp8uSYCjXOMnkUyb3CQ3Q",
	"UUSbdMXOI_3HGiFviLZO6kNA", "UUUaT_39o1x6qWjz7K2pWcgw", "UUDogdKl7t7NHzQ95aEwkdMw"
}

def getSubs(channelId):
	request = youtube.channels().list(part="statistics", id=channelId)
	channel = request.execute()

	return channel["items"][0]["statistics"]["subscriberCount"]

def getVideoInfo(videoId):
	request = youtube.videos().list(part="statistics,snippet", id=videoId)
	video = request.execute()

	item = video["items"][0]
	title = item["snippet"]["title"]
	views = item["statistics"]["viewCount"]
	category = item["snippet"]["categoryId"]
	channelId = item["snippet"]["channelId"]

	# Try to get best quality thumbnail first, then work downward
	thumbs = item["snippet"]["thumbnails"]
	thumb = None
	if "maxres" in thumbs:
		thumb = thumbs["maxres"]["url"]
	elif "standard" in thumbs:
		thumb = thumbs["standard"]["url"]
	elif "high" in thumbs:
		thumb = thumbs["high"]["url"]
	elif "medium" in thumbs:
		thumb = thumbs["medium"]["url"]
	else:
		thumb = thumbs["default"]["url"]
	
	return (title, thumb, views, category, channelId)

def saveThumbnail(url, fileName):
	thumb_data = requests.get(url).content
	with open(fileName, "wb") as writer:
		writer.write(thumb_data)

def getPlaylistInfo(playlistId, page):
	request = youtube.playlistItems().list(part="contentDetails", playlistId=playlistId, maxResults=50, pageToken=page)
	return request.execute()

def downloadPlaylists():
	# For formatting
	totalProcessed = 0
	maxTotal = len(playlists) * numPerPlaylist

	for playlistId in playlists:
		# This code is fairly shit, downloads stuff from a playlist
		nextPage = None
		processed = 0
		while True:
			# Go through each page of the playlist
			currentPage = getPlaylistInfo(playlistId, nextPage)

			# Go through each video on each page
			for video in currentPage["items"]:
				processed += 1
				totalProcessed+=1
				print(f"=== Processing {totalProcessed}/{maxTotal} ===")

				videoId = video["contentDetails"]["videoId"]
				title, thumb, views, category, channelId = getVideoInfo(videoId)
				print(title)

				# Skip over music videos
				if category == 10:
					print("Music video, skipping")
					continue

				subs = getSubs(channelId)
				saveThumbnail(thumb, f"dataset/{videoId}_{views}_{subs}.jpg")

				print(f"{views} views, {subs} subscribers ({int(views) / int(subs)})\n")
				
				if processed >= numPerPlaylist:
					break

			if processed >= numPerPlaylist:
				break

			# Go to next playlist page if possible
			if "nextPageToken" in currentPage:
				nextPage = currentPage["nextPageToken"]
			else:
				break

				

if __name__ == "__main__":

	# Make dataset folder for storing images
	if not os.path.exists("dataset"):
		os.mkdir("dataset")

	downloadPlaylists()
	