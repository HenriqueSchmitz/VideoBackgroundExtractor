import cv2
from video_background_extractor import VideoBackgroundExtractor


videoPath = "example.mp4"
video = cv2.VideoCapture(videoPath)
extractor = VideoBackgroundExtractor()

exampleFrameNumber = 3000
video.set(cv2.CAP_PROP_POS_FRAMES, exampleFrameNumber)
ret, exampleFrame = video.read()
cv2.imshow("Example Frame", exampleFrame)

extractor.loadVideo(video)

background = extractor.getBackgroundFrame()
cv2.imshow("Background", background)

frameDifference = extractor.getDifferenceToBackground(exampleFrame)
cv2.imshow("Example Difference to Background", frameDifference)

differenceIndex = extractor.getDifferenceIndex(exampleFrame)
print(differenceIndex)

noBackgroundFrame = extractor.removeBackground(exampleFrame)
cv2.imshow("Example Frame with Removed Background", noBackgroundFrame)
cv2.waitKey(0)