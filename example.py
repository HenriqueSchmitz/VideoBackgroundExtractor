import cv2
import subprocess
subprocess.run(["git", "clone", "https://www.github.com/HenriqueSchmitz/VideoBackgroundExtractor.git"])
from VideoBackgroundExtractor import VideoBackgroundExtractor

videoPath = "VideoBackgroundExtractor/example.mp4"
video = cv2.VideoCapture(videoPath)
extractor = VideoBackgroundExtractor()

exampleFrameNumber = 3000
video.set(cv2.CAP_PROP_POS_FRAMES, exampleFrameNumber)
ret, exampleFrame = video.read()
cv2.imshow(exampleFrame)

extractor.loadVideo(video)

background = extractor.getBackgroundFrame()
cv2.imshow(background)

frameDifference = extractor.getDifferenceToBackground(exampleFrame)
cv2.imshow(frameDifference)

differenceIndex = extractor.getDifferenceIndex(exampleFrame)
print(differenceIndex)

noBackgroundFrame = extractor.removeBackground(exampleFrame)
cv2.imshow(noBackgroundFrame)