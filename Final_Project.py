'''
import PySimpleGUI as pyg

#pyg.Window(title="balls", layout=[[]], margins=(100,100)).read()



text = pyg.popup_get_text('Enter your name', title="Textbox")
print ("You entered: ", text)

layout=[
    [pyg.Text('Length '),pyg.Input()],
    [pyg.Text('Girth '),pyg.Input()],
    [pyg.Text('Shaved? '),pyg.Input()],
    [pyg.OK(), pyg.Cancel(), pyg.Exit()]
]

window = pyg.Window('Form', layout)
while True:
   event, values = window.read()
   if event == pyg.WIN_CLOSED or event == 'Exit':
      break
   print (event, values)
window.close()
'''
import os
import PySimpleGUI as sg
layout=[
    [sg.Text(text='Facial Emotion Recognizer',
   font=('Arial Bold', 25),
   size=20,
   expand_x=True,
   justification='center')],
    [sg.Button("Emotion Detection", size=(10,5))],
    [sg.Button("New Emotion",size=(10,5))],
    [sg.Button("Train on Dataset",size=(10,5))],
    [sg.Exit(size=(10,5))]
]
window=sg.Window("FaceRecog",layout,size=(1000,500))
while True:
   event, values = window.read()
   if event=="New Emotion":
      import cv2
      import os
      cam = cv2.VideoCapture(0)
      cam.set(3, 640) # set video width
      cam.set(4, 480) # set video height
      face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
      face_id = sg.popup_get_text('Emotion:                    (Only Intger Value)') 
      sg.popup("Look at the camera")
      count = 0
      while(True):
         ret, img = cam.read()
         
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         faces = face_detector.detectMultiScale(gray, 1.3, 5)
         for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            
            cv2.imwrite("FaceDetectionProject/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
         k = cv2.waitKey(100) & 0xff 
         if k == 27:
            break
         elif count >= 10: 
               break
      sg.popup("Done!")
      cam.release()
      cv2.destroyAllWindows()
   if event=="Emotion Detection":

      import cv2
      import numpy as np
      import os 

      recognizer = cv2.face.LBPHFaceRecognizer_create()
      recognizer.read('FaceDetectionProject/trainer/trainer.yml')
      cascadePath = "haarcascade_frontalface_default.xml"
      faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

      font = cv2.FONT_HERSHEY_SIMPLEX

      id = 0


      names = ['None','Happy','Sad','Angry','Scared'] 


      cam = cv2.VideoCapture(0)
      cam.set(3, 640) # set video widht
      cam.set(4, 480) # set video height


      minW = 0.1*cam.get(3)
      minH = 0.1*cam.get(4)

      while True:

         ret, img =cam.read()
         

         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

         faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)))

         for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            
            if (confidence < 100):
                  id = names[id]
                  confidence = "  {0}%".format(round(100 - confidence))
            else:
                  id = "unknown"
                  confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
         
         cv2.imshow('camera',img) 

         k = cv2.waitKey(10) & 0xff 
         if k == 27:
            break


      print("\n Exit.")
      cam.release()
      cv2.destroyAllWindows()
   if event=='Train on Dataset':
      import cv2
      import numpy as np
      from PIL import Image
      import os
      path = 'FaceDetectionProject/dataset'
      recognizer = cv2.face.LBPHFaceRecognizer_create()
      detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");# function to get the images and label data
      def getImagesAndLabels(path):
         imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
         faceSamples=[]
         ids = []
         for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                  faceSamples.append(img_numpy[y:y+h,x:x+w])
                  ids.append(id)
         return faceSamples,ids
      sg.popup("\n Training faces. It will take a few seconds. Wait ...")
      faces,ids = getImagesAndLabels(path)
      recognizer.train(faces, np.array(ids))
      recognizer.write('FaceDetectionProject/trainer/trainer.yml') 
      sg.popup("\n {0} faces trained.".format(len(np.unique(ids))))
   print(event, values)
   if event in (None, 'Exit'):
      break
window.close()