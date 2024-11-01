import cv2
import os
import pickle
import face_recognition
import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{'databaseURL':"https://face-attendence-1e5e7-default-rtdb.firebaseio.com/"})
cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)
imgBackground = cv2.imread('Resources\\background.png')
#importing the mode images into a list
folderModepath = 'Resources\Modes'
modePathlist =os.listdir(folderModepath)
imgModelist =[]
for path in modePathlist:
    imgModelist.append(cv2.imread(os.path.join(folderModepath,path)))
    # print(len(imgModelist))
    
#load the encoding
print("loading encodefile....")
file=open("encodefile.p",'rb')
endoinglistwithIds= pickle.load(file)
file.close()
endcoingknown,studentIds= endoinglistwithIds
# print(studentIds)
print("encodefile loaded")
modetye =0
counter =0

while True:
    sucess,img = cap.read()
    
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #finding the encoding of the current frame face
    facecurrent = face_recognition.face_locations(imgs)
    encodecurrentface = face_recognition.face_encodings(imgs,facecurrent)
    for encodeface ,faceloc in zip(encodecurrentface,facecurrent):
        matches = face_recognition.compare_faces(endcoingknown,encodeface)#checking with the encoded faces
        facedis = face_recognition.face_distance(endcoingknown,encodeface)#cheking the value of matches
        # print("matches",matches)
        # print("facedis",facedis)
        matchindex = np.argmin(facedis)
        # print("match index",matchindex)
        if matches[matchindex]:
            # print("known Face deceted")
           
            id = studentIds[matchindex]
            # print(id)
            if counter==0:
                counter=1
                modetye=1
    if counter!=0:
        
        if counter==1:
            studentInfo = db.reference(f'student/{id}').get()#getting the student details form the data 
            #update data of attendence
            ref = db.reference(f'student/{id}')
            studentInfo['total_attendance']+=1
            ref.child('tota_attendance').set(studentInfo['total_attendance'])
            # print(studentInfo)
            cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            

            (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            
            offset = (414 - w) // 2
            
            cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
        counter+=1
        

    
    
    
    #setting the background images
    imgBackground[162:162+480 , 55:55+640]=img
    imgBackground[44:44+633 , 808:808+414]=imgModelist[modetye]
    #cv2.imshow("web cam",img)
    cv2.imshow("Face Attendence",imgBackground)
    cv2.waitKey(1)


