from .Types import Image, MonochromeImage, VideoCapture
from typing import List, Tuple
import numpy as np
import torch
import cv2

class VideoBackgroundExtractor:
  def __init__(self):
    self.__backgroundTensor = None;
    self.__isGpuAvailable = torch.cuda.is_available();

  def loadVideo(self, video: VideoCapture, numberOfFramesToUse: int = 25) -> None:
    framesTensor = self.__getRandomFramesTensorFromVideo(video, numberOfFramesToUse)
    median = torch.median(framesTensor, dim = 0)
    self.__backgroundTensor = median.values

  def loadVideoResiliently(self, video: VideoCapture, numberOfFramesToUse: int = 25, maximumMedianDifference: float = 0.1, maximumRetries: int = 10) -> None:
    framesTensor = self.__getRandomFramesTensorFromVideo(video, numberOfFramesToUse)
    median = torch.median(framesTensor, dim = 0)
    self.__backgroundTensor = median.values
    frameDifferencesTensor, medianDifference = self.__calculateFrameDifferences(framesTensor)
    numberOfRetries = 1
    while medianDifference.item() > maximumMedianDifference and numberOfRetries <= maximumRetries:
      numberOfRetries = numberOfRetries + 1
      goodFramesTensor = self.__filterFramesBelowMedian(framesTensor, frameDifferencesTensor, medianDifference)
      framesTensor = self.__completeTensorWithNewFrames(video, goodFramesTensor, numberOfFramesToUse)
      median = torch.median(framesTensor, dim = 0)
      self.__backgroundTensor = median.values
      frameDifferencesTensor, medianDifference = self.__calculateFrameDifferences(framesTensor)

  def isVideoCameraStatic(self, video: VideoCapture, numberOfFramesToUse: int = 25, maximumMedianDifference: float = 0.1) -> None:
    framesTensor = self.__getRandomFramesTensorFromVideo(video, numberOfFramesToUse)
    median = torch.median(framesTensor, dim = 0)
    self.__backgroundTensor = median.values
    frameDifferencesTensor, medianDifference = self.__calculateFrameDifferences(framesTensor)
    return medianDifference.item() <= maximumMedianDifference

  def __getRandomFramesTensorFromVideo(self, video: VideoCapture, numberOfFramesToUse: int) -> torch.Tensor:
    frames = self.__getRandomFramesFromVideo(video, numberOfFramesToUse)
    # Conversion to numpy array significantly improves performance for conversion to tensor
    framesTensor = torch.from_numpy(np.asarray(frames))
    if self.__isGpuAvailable:
      framesTensor = framesTensor.cuda()
    return framesTensor

  def __getRandomFramesFromVideo(self, video: VideoCapture, numberOfFramesToUse: int) -> List[Image]:
    previousPosition = video.get(cv2.CAP_PROP_POS_FRAMES)
    frameIds = torch.mul(torch.rand(numberOfFramesToUse), (video.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames = []
    for fid in frameIds:
        video.set(cv2.CAP_PROP_POS_FRAMES, fid.item())
        ret, frame = video.read()
        frames.append(frame)
    video.set(cv2.CAP_PROP_POS_FRAMES, previousPosition)
    return frames

  def __calculateFrameDifferences(self, framesTensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    frameDifferencesTensor = torch.tensor([self.__calculateDifferenceIndex(frameTensor) for frameTensor in framesTensor])
    medianDifference = torch.median(frameDifferencesTensor)
    return frameDifferencesTensor, medianDifference

  def __filterFramesBelowMedian(self, framesTensor: torch.Tensor, frameDifferencesTensor: torch.Tensor, medianDifference: torch.Tensor) -> torch.Tensor:
    mask = frameDifferencesTensor <= medianDifference.item()
    indices = torch.nonzero(mask).permute(1,0)[0]
    goodFramesTensor = framesTensor[indices]
    return goodFramesTensor

  def __completeTensorWithNewFrames(self, video: VideoCapture, goodFramesTensor: torch.Tensor, numberOfFramesToUse: int) -> torch.Tensor:
    amountOfGoodFrames = goodFramesTensor.size()[0]
    framesToGet = numberOfFramesToUse - amountOfGoodFrames
    newFramesTensor = self.__getRandomFramesTensorFromVideo(video, framesToGet)
    framesTensor = torch.cat((goodFramesTensor, newFramesTensor))
    return framesTensor
  
  def loadBackground(self, backgroundImage: Image) -> None:
    self.__backgroundTensor = torch.from_numpy(backgroundImage)
    if self.__isGpuAvailable:
      self.__backgroundTensor = self.__backgroundTensor.cuda()

  def getBackgroundFrame(self) -> Image:
    self.__validadeBackground()
    if self.__isGpuAvailable:
      return self.__backgroundTensor.cpu().numpy()
    return self.__backgroundTensor.numpy()
  
  def __validadeBackground(self) -> None:
    if self.__backgroundTensor == None:
      raise Exception("No video loaded to get background from")

  def getDifferenceToBackground(self, image: Image) -> MonochromeImage:
    self.__validadeBackground()
    if self.__isGpuAvailable:
      return self.__getDifferenceToBackgroundGpu(image)
    else:
      return self.__getDifferenceToBackgroundCpu(image)

  def __getDifferenceToBackgroundGpu(self, image: Image) -> MonochromeImage:
      imageTensor = torch.from_numpy(image).cuda().to(torch.int16)
      differenceTensor = self.__getDifferenceTensor(imageTensor)
      return differenceTensor.cpu().numpy()

  def __getDifferenceToBackgroundCpu(self, image: Image) -> MonochromeImage:
      imageTensor = torch.from_numpy(image).to(torch.int16)
      differenceTensor = self.__getDifferenceTensor(imageTensor)
      return differenceTensor.numpy()

  def __getDifferenceTensor(self, imageTensor: torch.Tensor) -> torch.Tensor:
      differencePerColorTensor = torch.abs(torch.subtract(imageTensor, self.__backgroundTensor))
      averageDifferenceTensor = torch.div(torch.sum(differencePerColorTensor, dim=2), 3)
      differenceTensor = averageDifferenceTensor.to(torch.uint8)
      return differenceTensor

  def getDifferenceIndex(self, image: Image) -> float:
    self.__validadeBackground()
    imageTensor = torch.from_numpy(image)
    if self.__isGpuAvailable:
      imageTensor = imageTensor.cuda()
    return self.__calculateDifferenceIndex(imageTensor)

  def __calculateDifferenceIndex(self, imageTensor: torch.Tensor) -> float:
    differenceTensor = self.__getDifferenceTensor(imageTensor.to(torch.int16))
    differenceMean = torch.div(torch.mean(differenceTensor.double()), 255)
    return differenceMean.item()

  def removeBackground(self, image: Image, thresholdFactor: float = 1, differenceIndexLimit: float = 0.2) -> Image:
    if self.__isGpuAvailable:
      imageTensor = torch.from_numpy(image).cuda().to(torch.int16)
    else:
      imageTensor = torch.from_numpy(image).to(torch.int16)
    thresholdedDifference = self.__getThresholdedDifferenceTensor(imageTensor, thresholdFactor, differenceIndexLimit)
    thresholdedImage = self.__applyThresholdsToImageTensor(imageTensor, thresholdedDifference).to(torch.uint8)
    if self.__isGpuAvailable:
      thresholdedImage = thresholdedImage.cpu()
    return thresholdedImage.numpy()

  def __getThresholdedDifferenceTensor(self, imageTensor: torch.Tensor, thresholdFactor: torch.Tensor, differenceIndexLimit: float) -> torch.Tensor:
    differenceTensor = self.__getDifferenceTensor(imageTensor)
    differenceMean = torch.mean(differenceTensor.double())
    differenceIndex = torch.div(differenceMean, 255)
    if differenceIndex <= differenceIndexLimit:
      differenceThreshold = torch.mul(differenceMean, thresholdFactor)
      thresholdedDifference = torch.where(differenceTensor > differenceThreshold, 1, 0)
    else:
      thresholdedDifference = torch.ones(differenceTensor.size())
      if self.__isGpuAvailable:
        thresholdedDifference = thresholdedDifference.cuda()
    return thresholdedDifference

  def __applyThresholdsToImageTensor(self, imageTensor: torch.Tensor, thresholdsTensor: torch.Tensor):
    reshapedImage = torch.permute(imageTensor, (2, 0, 1))
    reshapedThresholdedImage = torch.mul(reshapedImage, thresholdsTensor)
    thresholdedImage = torch.permute(reshapedThresholdedImage, (1, 2, 0))
    return thresholdedImage
