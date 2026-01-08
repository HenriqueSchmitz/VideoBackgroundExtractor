from .Types import Image, MonochromeImage
from typing import List, Tuple
import numpy as np
import torch
import cv2


class VideoBackgroundExtractor:
    def __init__(self, isGpuUseAllowed=True):
        """
        Parameters:
        - isGpuUseAllowed: allows the user to force this class to use cpu processing even if gpu is available
        """
        self.__backgroundTensor = None
        self.__device = "cpu"
        if isGpuUseAllowed:
            self.__device = "cuda" if torch.cuda.is_available() else self.__device
            self.__device = "mps" if torch.backends.mps.is_available() else self.__device

    def loadVideo(
        self,
        video: cv2.VideoCapture,
        numberOfFramesToUse: int = 25
    ) -> None:
        """
        Receives video and compiles background. Necessary for further operations.

        Parameters:
        - video: VideoCapture object from opencv containing a video taken from a fixed camera angle, 
        where the background is constant, but contains foreground objects that move.
        - numberOfFramesToUse: changes the number of random frames collected to compile background image.
        Increasing this can improve background image quality at the cost of loading performance.
        """
        framesTensor = self.__getRandomFramesTensorFromVideo(video, numberOfFramesToUse)
        median = torch.median(framesTensor, dim=0)
        self.__backgroundTensor = median.values

    def loadVideoResiliently(
        self,
        video: cv2.VideoCapture,
        numberOfFramesToUse: int = 25,
        maximumMedianDifference: float = 0.1,
        maximumRetries: int = 10
    ) -> None:
        """
        Receives video and compiles background. Necessary for further operations.
        This variation of the method is better suited for situations where there are too many bad frames, where the
        background is blocked or the camera view changed. This method will use an approximation to find the images with
        the most commonly appearing background and use only those for compiling.

        Parameters:
        - video: VideoCapture object from opencv containing a video taken from a fixed camera angle, 
        where the background is constant, but contains foreground objects that move
        - numberOfFramesToUse: changes the number of random frames collected to compile background image.
        Increasing this can improve background image quality at the cost of loading performance
        - maximumMedianDifference: determines how similar the frames must be to the background image to 
        be used as part of generating the final background. For situations where larger portions of the 
        background are covered by objects, this value might need to be increased
        - maximumRetries: maximum number of times the process will try to look for {numberOfFramesToUse} frames to find
        suitable frames for generating the background
        """
        framesTensor = self.__getRandomFramesTensorFromVideo(video, numberOfFramesToUse)
        median = torch.median(framesTensor, dim=0)
        self.__backgroundTensor = median.values
        frameDifferencesTensor, medianDifference = self.__calculateFrameDifferences(framesTensor)
        numberOfRetries = 1
        while medianDifference.item() > maximumMedianDifference and numberOfRetries <= maximumRetries:
            numberOfRetries = numberOfRetries + 1
            goodFramesTensor = self.__filterFramesBelowMedian(framesTensor, frameDifferencesTensor, medianDifference)
            framesTensor = self.__completeTensorWithNewFrames(video, goodFramesTensor, numberOfFramesToUse)
            median = torch.median(framesTensor, dim=0)
            self.__backgroundTensor = median.values
            frameDifferencesTensor, medianDifference = self.__calculateFrameDifferences(framesTensor)

    def isVideoCameraStatic(
        self,
        video: cv2.VideoCapture,
        numberOfFramesToUse: int = 25,
        maximumMedianDifference: float = 0.1
    ) -> bool:
        """
        Compiles a background for the video and compares it to random frames to determine if the provided video contains
        a constant background or not.

        Parameters:
        - video: VideoCapture object from opencv containing a video taken from a fixed camera angle, where the
        background is constant, but contains foreground objects that move
        - numberOfFramesToUse: changes the number of random frames collected to compile background image. Increasing
        this can improve background image quality at the cost of loading performance
        - maximumMedianDifference: determines how similar the frames must be to the background image for the video to be
        considered static. For situations where larger portions of the background are covered by objects, this value
        might need to be increased
        """
        framesTensor = self.__getRandomFramesTensorFromVideo(video, numberOfFramesToUse)
        median = torch.median(framesTensor, dim=0)
        self.__backgroundTensor = median.values
        _, medianDifference = self.__calculateFrameDifferences(framesTensor)
        return medianDifference.item() <= maximumMedianDifference

    def __getRandomFramesTensorFromVideo(
        self,
        video: cv2.VideoCapture,
        numberOfFramesToUse: int
    ) -> torch.Tensor:
        frames = self.__getRandomFramesFromVideo(video, numberOfFramesToUse)
        # Conversion to numpy array significantly improves performance for conversion to tensor
        framesTensor = torch.from_numpy(np.asarray(frames)).to(self.__device)
        return framesTensor

    def __getRandomFramesFromVideo(
        self,
        video: cv2.VideoCapture,
        numberOfFramesToUse: int
    ) -> List[Image]:
        previousPosition = video.get(cv2.CAP_PROP_POS_FRAMES)
        frameIds = torch.mul(torch.rand(numberOfFramesToUse), (video.get(cv2.CAP_PROP_FRAME_COUNT)))
        frames = []
        for fid in frameIds:
            video.set(cv2.CAP_PROP_POS_FRAMES, fid.item())
            _, frame = video.read()
            frames.append(frame)
        video.set(cv2.CAP_PROP_POS_FRAMES, previousPosition)
        return frames

    def __calculateFrameDifferences(self, framesTensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frameDifferencesTensor = torch.tensor([self.__calculateDifferenceIndex(frameTensor)
                                              for frameTensor in framesTensor])
        medianDifference = torch.median(frameDifferencesTensor)
        return frameDifferencesTensor, medianDifference

    def __filterFramesBelowMedian(
        self,
        framesTensor: torch.Tensor,
        frameDifferencesTensor: torch.Tensor,
        medianDifference: torch.Tensor
    ) -> torch.Tensor:
        mask = frameDifferencesTensor <= medianDifference.item()
        indices = torch.nonzero(mask).permute(1, 0)[0]
        goodFramesTensor = framesTensor[indices]
        return goodFramesTensor

    def __completeTensorWithNewFrames(
        self,
        video: cv2.VideoCapture,
        goodFramesTensor: torch.Tensor,
        numberOfFramesToUse: int
    ) -> torch.Tensor:
        amountOfGoodFrames = goodFramesTensor.size()[0]
        framesToGet = numberOfFramesToUse - amountOfGoodFrames
        newFramesTensor = self.__getRandomFramesTensorFromVideo(video, framesToGet)
        framesTensor = torch.cat((goodFramesTensor, newFramesTensor))
        return framesTensor

    def loadBackground(self, backgroundImage: Image) -> None:
        """
        Loads previously known video background. This can be used instead of loadVideo to provide background necessary
        for other operations in a faster or more controlled manner.

        Parameters:
        - backgroundImage: image to be used as the background layer for comparison in other operations
        """
        self.__backgroundTensor = torch.from_numpy(backgroundImage).to(self.__device)

    def getBackgroundFrame(self) -> Image:
        """
        Returns background frame calculated based on the provided video.
        """
        self.__validateBackground()
        return self.__backgroundTensor.to(torch.uint8).cpu().numpy()

    def __validateBackground(self) -> None:
        if self.__backgroundTensor == None:
            raise Exception("No video loaded to get background from")

    def getDifferenceToBackground(self, image: Image) -> MonochromeImage:
        """
        Creates a single channel (monochrome) image, where each pixel's itensity represents how different the provided
        image is at that point compared to the video background. Here 0 represents a pixel completely equal to the
        background an 255 a pixel completely different to the background.

        Parameters:
        - image: frame from the video to be compared with the compiled background image
        """
        self.__validateBackground()
        imageTensor = torch.from_numpy(image).to(self.__device).to(torch.int16)
        differenceTensor = self.__getDifferenceTensor(imageTensor)
        return differenceTensor.to(torch.uint8).cpu().numpy()

    def __getDifferenceTensor(self, imageTensor: torch.Tensor) -> torch.Tensor:
        differencePerColorTensor = torch.abs(torch.subtract(imageTensor, self.__backgroundTensor))
        averageDifferenceTensor = torch.div(torch.sum(differencePerColorTensor, dim=2), 3)
        differenceTensor = averageDifferenceTensor.to(torch.uint8)
        return differenceTensor

    def getDifferenceIndex(self, image: Image) -> float:
        """
        Calculates the average pixel value of the getDifferenceToBackground image. This represents how different a given
        frame is to the video background. This can be used to identify moments in the video where the camera view
        changes or the camera is blocked by some object. Here 0 represents a frame being exactly equal to the background
        and 1 being completely different.

        Parameters:
        - image: frame from the video to be compared with the compiled background image
        """
        self.__validateBackground()
        imageTensor = torch.from_numpy(image).to(self.__device)
        return self.__calculateDifferenceIndex(imageTensor)

    def __calculateDifferenceIndex(self, imageTensor: torch.Tensor) -> float:
        differenceTensor = self.__getDifferenceTensor(imageTensor.to(torch.int16))
        differenceMean = torch.div(torch.mean(differenceTensor.double()), 255)
        return differenceMean.item()

    def removeBackground(
        self,
        image: Image,
        thresholdFactor: float = 2,
        differenceIndexLimit: float = 0.2
    ) -> Image:
        """
        Compares an image to the compiled background and generates a new image containing only objects that are not part
        of the background. Anything that is part of the background is changed to black.

        Parameters:
        - image: frame from the video to be compared with the compiled background image
        - thresholdFactor: the amount of times the difference of a pixel has to be higher than the average difference to
        the background to be compared part of the foreground. Higher values will cause only stronger differences to be
        considered part of the foreground. Can be used to decrease noise in resulting image or to ensure more of the
        foreground is detected
        - differenceIndexLimit: how different a frame can be to the background as a whole for this process to be
        applied. If an image is more different than this index, it will be considered as containing a different camera
        view or background and will be returned unchanged
        """
        imageTensor = torch.from_numpy(image).to(self.__device).to(torch.int16)
        thresholdedDifference = self.__getThresholdedDifferenceTensor(
            imageTensor, thresholdFactor, differenceIndexLimit)
        thresholdedImage = self.__applyThresholdsToImageTensor(imageTensor, thresholdedDifference).to(torch.uint8)
        return thresholdedImage.to(torch.uint8).cpu().numpy()

    def __getThresholdedDifferenceTensor(
        self,
        imageTensor: torch.Tensor,
        thresholdFactor: torch.Tensor,
        differenceIndexLimit: float,
        isZerosWhenDifferenceLimitExceeded: bool = False
    ) -> torch.Tensor:
        differenceTensor = self.__getDifferenceTensor(imageTensor)
        differenceMean = torch.mean(differenceTensor.double())
        differenceIndex = torch.div(differenceMean, 255)
        if differenceIndex <= differenceIndexLimit:
            differenceThreshold = torch.mul(differenceMean, thresholdFactor)
            thresholdedDifference = torch.where(differenceTensor > differenceThreshold, 1, 0)
        else:
            if isZerosWhenDifferenceLimitExceeded:
                thresholdedDifference = torch.zeros(differenceTensor.size()).to(self.__device)
            else:
                thresholdedDifference = torch.ones(differenceTensor.size()).to(self.__device)
        return thresholdedDifference

    def __applyThresholdsToImageTensor(self, imageTensor: torch.Tensor, thresholdsTensor: torch.Tensor):
        reshapedImage = torch.permute(imageTensor, (2, 0, 1))
        reshapedThresholdedImage = torch.mul(reshapedImage, thresholdsTensor)
        thresholdedImage = torch.permute(reshapedThresholdedImage, (1, 2, 0))
        return thresholdedImage

    def grayOutBackground(
        self,
        image: Image,
        thresholdFactor: float = 2,
        differenceIndexLimit: float = 0.2
    ) -> Image:
        """
        Compares an image to the compiled background and generates a new image where anything that is part of the
        background is in grayscale.

        Parameters:
        - image: frame from the video to be compared with the compiled background image
        - thresholdFactor: the amount of times the difference of a pixel has to be higher than the average difference to
        the background to be compared part of the foreground. Higher values will cause only stronger differences to be 
        considered part of the foreground. Can be used to decrease noise in resulting image or to ensure more of the
        foreground is detected
        - differenceIndexLimit: how different a frame can be to the background as a whole for this process to be 
        applied. If an image is more different than this index, it will be considered as containing a different camera
        view or background and will be returned fully as grayscale
        """
        imageTensor = torch.from_numpy(image).to(self.__device).to(torch.int16)
        grayscaleImage = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        grayscaleTensor = torch.from_numpy(grayscaleImage).to(self.__device).to(torch.int16)
        thresholdedDifference = self.__getThresholdedDifferenceTensor(
            imageTensor, thresholdFactor, differenceIndexLimit, isZerosWhenDifferenceLimitExceeded=True)
        foregroundPart = self.__applyThresholdsToImageTensor(imageTensor, thresholdedDifference)
        grayscalePart = self.__applyThresholdsToImageTensor(grayscaleTensor, 1 - thresholdedDifference)
        finalImage = torch.add(foregroundPart, grayscalePart)
        return finalImage.to(torch.uint8).cpu().numpy()
