"""
Extraire les images d'une vidéo, les redécouper et les resize.
"""

# Imports
import cv2
import sys
import numpy as np

# Constants
EACH_FRAME = 10
TARGET_WIDTH = 600
TARGET_HEIGHT = 600
RATIO_WIDTH = 0.5
RATIO_HEIGHT = 0.0 # top = 0.0, bottom = 1.0 - 0.0

# 1920 x 1080 -> 600 x 600 en raccoursisant par la gauche et la droite équitablement puis par le bas

def processFrame(frame, frameNumber):
    """
    Process a frame and save it as a .bmp file.
    """

    # Compute the offset to remove from each part of the frame
    width, height = len(frame[0]), len(frame)
    overflowW, overflowH = width - TARGET_WIDTH, height - TARGET_HEIGHT
    minusTop = overflowH * RATIO_HEIGHT
    minusBottom = overflowH * (1.0 - RATIO_HEIGHT)
    minusLeft = overflowW * RATIO_WIDTH
    minusRight = overflowW * (1.0 - RATIO_WIDTH)

    # Cut the frame
    image = frame[minusTop:height-minusBottom][minusLeft:width-minusRight]

    print(type(frame))
    # cv2.imwrite("senna/senna_frame%d.jpg" % count, image)     # save frame as JPEG file


def main():
    """
    Process a video file.
    """
    videoFileName = sys.argv[0]
    print("Reading file %s" % videoFileName)
    video = cv2.VideoCapture(videoFileName)
    success = True
    frameNumber = 0

    while success:
        success, frame = video.read()
        frameNumber += 1

        if frameNumber % EACH_FRAME == 0:
            print("Processing frame %d ..." % frameNumber)
            processFrame(frame, i)

    video.release()

if __name__ == "__main__":
    main()
