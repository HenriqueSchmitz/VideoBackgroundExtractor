import numpy as np
import torch
import cv2

class VideoBackgroundExtractor:
  def __init__(self):
    self.__backgroundTensor = None;
    self.__isGpuAvailable = torch.cuda.is_available();

  def loadVideo(self, video, numberOfFramesToUse = 25):
    frames = self.__getRandomFramesFromVideo(video, numberOfFramesToUse)
    framesTensor = torch.from_numpy(np.asarray(frames))
    if self.__isGpuAvailable:
      framesTensor = framesTensor.cuda()
    median = torch.median(framesTensor, dim = 0)
    self.__backgroundTensor = median.values

  def __getRandomFramesFromVideo(self, video, numberOfFramesToUse):
    previousPosition = video.get(cv2.CAP_PROP_POS_FRAMES)
    frameIds = torch.mul(torch.rand(numberOfFramesToUse), (video.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames = []
    for fid in frameIds:
        video.set(cv2.CAP_PROP_POS_FRAMES, fid.item())
        ret, frame = video.read()
        frames.append(frame)
    video.set(cv2.CAP_PROP_POS_FRAMES, previousPosition)
    return frames
  
  def loadBackground(self, backgroundImage):
    self.__backgroundTensor = torch.from_numpy(backgroundImage)
    if self.__isGpuAvailable:
      self.__backgroundTensor = self.__backgroundTensor.cuda()

  def getBackgroundFrame(self):
    self.__validadeBackground()
    if self.__isGpuAvailable:
      return self.__backgroundTensor.cpu().numpy()
    return self.__backgroundTensor.numpy()
  
  def __validadeBackground(self):
    if self.__backgroundTensor == None:
      raise Exception("No video loaded to get background from")

  def getDifferenceToBackground(self, image):
    self.__validadeBackground()
    if self.__isGpuAvailable:
      return self.__getDifferenceToBackgroundGpu(image)
    else:
      return self.__getDifferenceToBackgroundCpu(image)

  def __getDifferenceToBackgroundGpu(self, image):
      imageTensor = torch.from_numpy(image).cuda().to(torch.int16)
      differenceTensor = self.__getDifferenceTensor(imageTensor)
      return differenceTensor.cpu().numpy()

  def __getDifferenceToBackgroundCpu(self, image):
      imageTensor = torch.from_numpy(image).to(torch.int16)
      differenceTensor = self.__getDifferenceTensor(imageTensor)
      return addedFrame.numpy()

  def __getDifferenceTensor(self, imageTensor):
      differencePerColorTensor = torch.abs(torch.subtract(imageTensor, self.__backgroundTensor))
      averageDifferenceTensor = torch.div(torch.sum(differencePerColorTensor, dim=2), 3)
      differenceTensor = averageDifferenceTensor.to(torch.uint8)
      return differenceTensor

  def getDifferenceIndex(self, image):
    self.__validadeBackground()
    imageTensor = torch.from_numpy(image)
    if self.__isGpuAvailable:
      imageTensor = imageTensor.cuda()
    differenceTensor = self.__getDifferenceTensor(imageTensor.to(torch.int16))
    differenceMean = torch.div(torch.mean(differenceTensor.double()), 255)
    return differenceMean.item()

  def removeBackground(self, image, thresholdFactor = 1, differenceIndexLimit = 0.2):
    if self.__isGpuAvailable:
      imageTensor = torch.from_numpy(image).cuda().to(torch.int16)
    else:
      imageTensor = torch.from_numpy(image).to(torch.int16)
    thresholdedDifference = self.__getThresholdedDifferenceTensor(imageTensor, thresholdFactor, differenceIndexLimit)
    thresholdedImage = self.__applyThresholdsToImageTensor(imageTensor, thresholdedDifference).to(torch.uint8)
    if self.__isGpuAvailable:
      thresholdedImage = thresholdedImage.cpu()
    return thresholdedImage.numpy()

  def __getThresholdedDifferenceTensor(self, imageTensor, thresholdFactor, differenceIndexLimit):
    differenceTensor = self.__getDifferenceTensor(imageTensor)
    differenceMean = torch.mean(differenceTensor.double())
    differenceIndex = torch.div(differenceMean, 255)
    if differenceIndex <= differenceIndexLimit:
      differenceThreshold = torch.mul(differenceMean, thresholdFactor)
      thresholdedDifference = torch.where(differenceTensor > differenceMean, 1, 0)
    else:
      thresholdedDifference = torch.ones(differenceTensor.size())
      if self.__isGpuAvailable:
        thresholdedDifference = thresholdedDifference.cuda()
    return thresholdedDifference

  def __applyThresholdsToImageTensor(self, imageTensor, thresholdsTensor):
    reshapedImage = torch.permute(imageTensor, (2, 0, 1))
    reshapedThresholdedImage = torch.mul(reshapedImage, thresholdsTensor)
    thresholdedImage = torch.permute(reshapedThresholdedImage, (1, 2, 0))
    return thresholdedImage
