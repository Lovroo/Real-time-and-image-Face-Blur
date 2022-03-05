from pyimagesearch.face_blurring import anonimiziraj_obraz_pixli
from pyimagesearch.face_blurring import anonimen_obraz_simple
import numpy as np
import argparse
import cv2

# sestavi argument in ga razčleni
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, #Za import slike
	help="pot do slike")
ap.add_argument("-f", "--face", required=True, #Za detektor obraza (algoritem)
	help="pot do detektorja obrazov")
ap.add_argument("-m", "--method", type=str, default="simple", #izberemo lahko simple blur alpa pixiliran blur
	choices=["simple", "pixelated"],
	help="metoda bluranja obrazov")
ap.add_argument("-b", "--blocks", type=int, default=20, #nivo anonimizacije nastavimo s tem argumentom
	help="# blokov, ki jih bomo blurali")
ap.add_argument("-c", "--confidence", type=float, default=0.5, #Natavimo koliko je %-program prepričan, da je zaznan obraz
	help="najmanjša verjetnost, da je obraz zaznan")
args = vars(ap.parse_args())

# naložimo model za zaznavanje obrazov
print("[INFO] Nalagam model za zaznavanje obrazov...")
prototxtPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
weightsPath = "./face_detector/deploy.prototxt.txt"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
# naložimo sliko z diska, jo podvojimo, dobimo vse podatke ter dimenzije
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]
#Iz slike naredimo blob
blob = cv2.dnn.blobFromImage(image, 1.0, (256, 256),
	(104.0, 177.0, 123.0))
#Blob gre čez detekcijo obraza
print("[INFO] Zamegljujem obraze...")
net.setInput(blob)
detections = net.forward()
# loop čez detekcije
for i in range(0, detections.shape[2]):
	# dobimo  koliko % je prepričanja, da je obraz zaznan
	confidence = detections[0, 0, i, 2]
	# izbriši manj veretne detekcije in preveri, da je verjetnost večja od minimalni, ki smo jo nastavili v argumentu
	if confidence > args["confidence"]:
		# zračunaj x in y kordinate kjer je zaznan obraz
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# ekstrahiraj ROI obraza
		face = image[startY:endY, startX:endX]
# preveri če imamo izbrano simple metodo za bluranje obrazov
		if args["method"] == "simple":
			face = anonimen_obraz_simple(face, factor=3.0)
# Drugače uporabi pixel metodo za blur
		else:
			face = anonimiziraj_obraz_pixli(face, blocks=args["blocks"])
# Shrani zablurano sliko
image[startY:endY, startX:endX] = face
# Prikaži originalno sliko ter obdelano sliko
output = np.hstack([orig, image])
output = cv2.resize(output, (960, 540))
cv2.imshow("Output", output)
cv2.waitKey(0)
