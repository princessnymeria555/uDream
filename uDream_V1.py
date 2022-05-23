import cv2
import math
import time
import mediapipe as mp 


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

parpadeo = False
conteo = 0
tiempo = 0
inicio = 0
final = 0
conteo_sue = 0
muestra = 0

mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness = 1,circle_radius = 1)

mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces = 1)

while True:
    ret,frame=cap.read()
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    resultados = MallaFacial.process(frameRGB)
    px=[]
    py=[]
    lista=[]
    r=5
    t=3

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame,rostros,mpMallaFacial.FACEMESH_CONTOURS,ConfDibu,ConfDibu)

            #ahora vamos a extraer los puntos del rotro detectado
            for id,puntos in enumerate(rostros.landmark):
                #print(puntos) #Nos entrega una proporcion
                al,an,c=frame.shape
                x,y=int(puntos.x*an), int(puntos.y*al)
                px.append(x)
                py.append(y)
                lista.append([id,x,y])

                if len(lista)==468:
                    #ojo derecho
                    x1,y1=lista[145][1:]
                    x2,y2=lista[159][1:]
                    cx,cy=(x1*x2)//2,(y1*y2)//2
                    cv2.line(frame,(x1,y1),(x2,y2),(0,0,0),t)
                    cv2.circle(frame,(x1,y1),r,(0,0,0),cv2.FILLED)
                    cv2.circle(frame,(x2,y2),r,(0,0,0),cv2.FILLED)
                    cv2.circle(frame,(cx,cy),r,(0,0,0),cv2.FILLED)
                    longitud1=math.hypot(x2-x1,y2-y1)
                    print(longitud1)

                    #ojo izquierdo
                    x3,y3=lista[374][1:]
                    x4,y4=lista[386][1:]
                    cx2,cy2=(x3*x4)//2,(y3*y4)//2
                    '''
                    cv2.line(frame,(x1,x2),(x2,y2),(0,0,0),t)
                    cv2.circle(frame,(x1,y1),r,(0,0,0),cv2.FILLED)
                    cv2.circle(frame,(x2,y2),r,(0,0,0),cv2.FILLED)
                    cv2.circle(frame,(cx,cy),r,(0,0,0),cv2.FILLED)
                    '''
                    longitud2=math.hypot(x4-x3,y4-y3)
                    print(longitud2)
                    
                    #Conteo de parpadeo
                    cv2.putText(frame,f'Parpadeos: {int(conteo)}',(60,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                    cv2.putText(frame,f'Micro Sueno: {int(conteo_sue)}',(300,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    cv2.putText(frame,f'Duracion: {str(muestra)}',(400,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                    if longitud1<=11.5 and longitud2 <= 11.5 and parpadeo == False:
                        conteo=conteo+1
                        parpadeo=True
                        inicio=time.time()
                    elif longitud2 > 16 and longitud2 > 16 and parpadeo == True:
                        parpadeo=False
                        final = time.time()
                    print(conteo)

                    #temporizador
                    tiempo=round(final-inicio,0)

                    #contador de micro sueños
                    if tiempo >=3:
                        conteo_sue=conteo_sue+1
                        muestra=tiempo
                        inicio=0
                        final=0


    
    cv2.imshow('webCam',frame)
    if (cv2.waitKey(1) == ord('s')):
        break

cap.release()
cv2.destroyAllWindows()

'''
    cv2.imshow('m',cap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
