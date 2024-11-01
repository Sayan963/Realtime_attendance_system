import cv2
import face_recognition
import pickle
import os
#importing the student images into a list
folderpath = 'Images'
Pathlist =os.listdir(folderpath)
imglist =[]
studentIds=[]
for path in Pathlist:
    imglist.append(cv2.imread(os.path.join(folderpath,path)))
    studentIds.append(os.path.splitext(path)[0])#spliting the student id from the images and stored in the studentids
# print(studentIds)
# print(len(imgModelist)
def findencoding(imageslist):
    encodelist=[]
    for img in imageslist:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#converting to RGB as face_recognition uses RGB
        encode = face_recognition.face_encodings(img)[0]#doing encofing using face_endoing
        encodelist.append(encode)
    return encodelist
print("Encoding started....")
endcoingknown = findencoding(imglist)
endoinglistwithIds = [endcoingknown,studentIds]
# print(endcoingknown)
print("Encoding comleted")
#saving the as encodefile.p for furtther uses
file = open("encodefile.p",'wb')
pickle.dump(endoinglistwithIds,file)
file.close()
print("File is saved")