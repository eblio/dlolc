import os
import cv2

DOIT = False
PATH = "../data"
EXTENSION = ".bmp"
OUTPUT_SIZE = (128, 128)

def main():
    for subdir, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(EXTENSION):
                path = subdir + "/" + file
                image = cv2.imread(path)
                output = cv2.resize(image, OUTPUT_SIZE)
                cv2.imwrite(path, output)

if __name__ == "__main__" and DOIT: # Sécurite pour ne pas lancer un traitement inintentionellement
    main()
