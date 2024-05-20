from typing import List, NewType
import cv2

Image = List[List[List[int]]]
MonochromeImage = List[List[int]]
VideoCapture = NewType("VideoCapture", cv2.VideoCapture)