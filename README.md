# VideoBackgroundExtractor
Library to extract background from static video. 

For usage with general python, example.py can be used as reference.

For usage with google colab, the following notebook can be used: https://colab.research.google.com/drive/1hMrAC0tIAhoO3theECk0OMmtR-JB-UsE?usp=sharing

Getting example image:

```
import cv2
video = cv2.VideoCapture(example.mp4)
video.set(cv2.CAP_PROP_POS_FRAMES, 3000)
ret, exampleFrame = video.read()
```

**Original Image:**
![Original Image](https://raw.githubusercontent.com/HenriqueSchmitz/VideoBackgroundExtractor/blob/main/samples/originalImage.png)

Getting background from video:

```
extractor = VideoBackgroundExtractor()
extractor.loadVideo(video)
background = extractor.getBackgroundFrame()
```

**Background Extracted from Video:**
![Background from Video](https://raw.githubusercontent.com/HenriqueSchmitz/VideoBackgroundExtractor/blob/main/samples/background.png)

Getting image with difference to background:

```
frameDifference = extractor.getDifferenceToBackground(exampleFrame)
```

**Difference to Background:**
![Difference to Background](https://raw.githubusercontent.com/HenriqueSchmitz/VideoBackgroundExtractor/blob/main/samples/differenceToBackground.png)

Getting example image with background removed:

```
noBackgroundFrame = extractor.removeBackground(exampleFrame)
```

**Image with no Background:**
![No Background Image](https://raw.githubusercontent.com/HenriqueSchmitz/VideoBackgroundExtractor/blob/main/samples/noBackgroundImage.png)

## Distribution

To create a wheel for distribution, run:
```
pip wheel .
```