"""
Extraire les images d'une vidéo, les redécouper et les resize.
"""

# Imports
import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

# Constants
EACH_FRAME = 10
TARGET_WIDTH = 600
TARGET_HEIGHT = 600
RATIO_WIDTH = 0.5
RATIO_HEIGHT = 0.0 # top = 0.0, bottom = 1.0 - 0.0

PATH = "../data/%s/"
EXTENSION = ".bmp"
FILENAME = PATH + "%s_%d_%dx%d" + EXTENSION

# 1920 x 1080 -> 600 x 600 en raccoursisant par la gauche et la droite équitablement puis par le bas

def processFrame(frame, frameNumber, fileName):
    """
    Process a frame and save it as a .bmp file.
    """

    print("Processing frame %d ..." % (frameNumber))

    # Compute the offset to remove from each part of the frame
    width, height = len(frame[0]), len(frame)
    overflowW, overflowH = width - TARGET_WIDTH, height - TARGET_HEIGHT
    minusTop = int(overflowH * RATIO_HEIGHT)
    minusBottom = int(overflowH * (1.0 - RATIO_HEIGHT))
    minusLeft = int(overflowW * RATIO_WIDTH)
    minusRight = int(overflowW * (1.0 - RATIO_WIDTH))

    # Cut the frame
    image = frame[minusTop:height-minusBottom, minusLeft:width-minusRight]

    # Save the image
    path = FILENAME % (fileName, fileName, frameNumber, TARGET_HEIGHT, TARGET_WIDTH)
    cv2.imwrite(path, image)


def main():
    """
    Process a video file.
    """

    # Get the command lien arguments
    videoFileName = sys.argv[1]
    name = sys.argv[2]

    # Create the output directory
    try:
        os.mkdir(PATH % (name))
    except:
        print("Répertoire déjà existant")

    # Open the video file
    video = cv2.VideoCapture(videoFileName)
    success, frame = video.read()
    frameNumber = 1

    print("Reading file %s" % (videoFileName))

    while success:
        if frameNumber % EACH_FRAME == 0:
            processFrame(frame, frameNumber, name)

        success, frame = video.read()
        frameNumber += 1

    video.release()
    print("Finished")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main()
    else:
        print("Nécessite 2 arguments : [path vidéo] [nom fichier sortie]")
