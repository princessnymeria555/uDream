import cv2
import os
import imutils
Name=input("Ingrese el nombre: ")
dataPath ='C:/Users/79449/Documents/PROYECTO MICH/datasheet'
NamesPath = dataPath + '/' + Name

if not os.path.exists(NamesPath):
	print('Carpeta creada: ',NamesPath)
	os.makedirs(NamesPath)
print("")
print("1. Left Eye")
print("2. Right Eye")
print("3. From Eye")
print("4. Without Eye")

Selec=int(input("Ingrese su eleción: "))

####cambiar segun el tipo de datashet

if(Selec==1):
	Eyes = 'Left_eye'
elif(Selec==2):
	Eyes = 'Right_eye'
elif(Selec==3):
	Eyes = 'From_eye'	
elif(Selec==4):
	Eyes = 'Without_eye'

EyesPath = NamesPath + '/' + Eyes

if not os.path.exists(EyesPath):
	print('Carpeta creada: ',EyesPath)
	os.makedirs(EyesPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:

	ret, frame = cap.read()
	if ret == False: break
	frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(EyesPath + '/rotro_{}.jpg'.format(count),rostro)
		count = count + 1
	cv2.imshow('frame',frame)

	k =  cv2.waitKey(1)
	if k == 27 or count >= 30:
		break

cap.release()
cv2.destroyAllWindows()