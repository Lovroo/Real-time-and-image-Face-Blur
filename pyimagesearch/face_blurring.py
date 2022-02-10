import numpy as np
import cv2

def anonimen_obraz_simple(image, factor =3.0):
    #Avtomatsko nastavi velikost blurra glede na velikost slike
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # preveri, da je širina kernela soda
    if kW % 2 == 0:
        kW -= 1
    # preveri, da je višina kernela soda
    if kH % 2 == 0:
        kH -= 1
    # Na sliko, ki jo vnesemo dodamo GaussianBlur
    return cv2.GaussianBlur(image, (kW, kH), 0)

def anonimiziraj_obraz_pixli(image, blocks=3):
    #razdeli sliko v NxN kvadrate
    (h, w) = image.shape[:2]
    xKoraki = np.linspace(0, w, blocks + 1, dtype="int")
    yKoraki = np.linspace(0, h, blocks + 1, dtype="int")
    # zanka čez vse x korake in vse y korake
    for i in range(1, len(yKoraki)):
        for j in range(1, len(xKoraki)):
            # Zapiši vse x in y kordinate za naš block
            startX = xKoraki[j - 1]
            startY = yKoraki[i - 1]
            endX = xKoraki[j]
            endY = yKoraki[i]
            # izvozi ROI z NumPy in izreži kvadrate, nariši kvadrat okoli ROI
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # Vrni blurano sliko
    return image


