from pytube import Playlist
from pytube.exceptions import AgeRestrictedError

pl = Playlist("https://youtube.com/playlist?list=PLr9w0uxRdNys6XuG681c5LKnqVg0KYf8L&si=uBL4OHB99MMLS0Dj")


for video in pl.videos:
    try:
        video.streams.first().download('/Users/jeongjanghun/Desktop/FacialRecognitionProject apr 13/test data/video data')
    except AgeRestrictedError:
        print(f"Skipping age-restricted video: {video.title}")