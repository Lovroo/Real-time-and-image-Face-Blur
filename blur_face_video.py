from pyimagesearch.face_blurring import anonimiziraj_obraz_pixli
from pyimagesearch.face_blurring import anonimen_obraz_simple
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# sestavi argument in ga razčleni
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,  # Za detektor obraza (algoritem)
                help="pot do detektorja obrazov")
ap.add_argument("-m", "--method", type=str, default="simple",  # izberemo lahko simple blur alpa pixiliran blur
                choices=["simple", "pixelated"],
                help="metoda bluranja obrazov")
ap.add_argument("-b", "--blocks", type=int, default=20,  # nivo anonimizacije nastavimo s tem argumentom
                help="# blokov, ki jih bomo blurali")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                # Natavimo koliko je %-program prepričan, da je zaznan obraz
                help="najmanjša verjetnost, da je obraz zaznan")
args = vars(ap.parse_args())

# naložimo model za zaznavanje obrazov
print("[INFO] Nalagam model za zaznavanje obrazov...")
prototxtPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
weightsPath = "./face_detector/deploy.prototxt.txt"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Zaženi video stream in počakaj, da se kamera  zažene
print("[INFO] začenjam prenašanje kamere...")
vs = VideoStream(src=2).start()
time.sleep(3.0)
# loop zanka čez vse frame, ki pridejo iz kamere
while True:
    # Zagrabi okvir iz kamere in mu daj največjo širino 400px
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # pridobi dimenzije okvirja in iz njega ustvari blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    # pošli blob čez omrežje in pridobi zaznan obraz
    net.setInput(blob)
    detections = net.forward()
    # loop zanka čez vse zaznane slike
    for i in range(0, detections.shape[2]):
        # pridobi zanesljivost detekcije
        confidence = detections[0, 0, i, 2]
        # izbriši slabe detekcije tako, da preverimo da je konfidenca večja, kot tista, ki smo jo nastavili
        if confidence > args["confidence"]:
            # izračunaj kordinate za kvadrakt objekta
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # izvozi obraz
            face = frame[startY:endY, startX:endX]
            # preveri kakšno metodo uporabljamo za zaznavsnje obraza
            if args["method"] == "simple":
                face = anonimen_obraz_simple(face, factor=3.0)
            else:
                face = anonimiziraj_obraz_pixli(face,
                                                blocks=args["blocks"])
            # shrani bluran obraz v novi sliki
            frame[startY:endY, startX:endX] = face
        # prikaži izhodno sliko
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # če kliknemo prekinemo zanko
        if key == ord("q"):
            break
