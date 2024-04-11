import cv2
import threading
from deepface import DeepFace

video_capture = cv2.VideoCapture(0)

#list of names in which we'll save the names of the faces in the frame, those names will be updated every 30 frames
names = [""]*20

#so we dont check for an already existing face in the frame multiple times
checked_faces = []


counter = 0
referenceImgk = cv2.imread("referenceImgk.jpg")
soulaymanImg = cv2.imread("soulaymanImg.jpg")
################################################################################################################################################################
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_bounding_box_with_names(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    global names, checked_faces

    #reinitializing the variable to update it with the names of the faces currently in the frame
    names = [""]*20

    #to itereate over names
    i = 0


    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        #Cutting the part of the frame containing the face
        face_roi = vid[y:y + h, x:x + w]

        #Checking for face owner in the cut frame
        name = check_face(face_roi)

        #Adding face name to global names list
        names[i] = name

        #Writing the name on screen
        if name == "no match":
            cv2.putText(vid, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.005 * h, (0, 0, 255), 3)
        else:
            cv2.putText(vid, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.005*h, (0, 255, 0), 1)  ############
        i += 1
    #Reinitializing the variable, check check_face to see use
    checked_faces = []
    return faces, names


#For optimizing we use this function to draw rectangles and names using the data we get from detect_bounding_box_with_names every 30 frames
def detect_bounding_box(vid):
    global names
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    i = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        if names[i] == "no match":
            cv2.putText(vid, names[i], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.005 * h, (0, 0, 255), 3)
        else:
            cv2.putText(vid, names[i], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.005*h, (0, 255, 0), 1)  ############
        i += 1

    return faces

#################################################################################################################################################################


def check_face(frame):
    global checked_faces
    who = " "
    try:
        #verify if we already checked khadija in the frame and found her ,
        # because there can only be one khadija in the picture and there's no point checking twice
        if not ("khadija" in checked_faces) and DeepFace.verify(frame, referenceImgk.copy())['verified']:
            who = "khadija"

            #Adding khadija to the checked faces list so we dont check for her again this iteration
            checked_faces.append(who)

        elif not ("soulayman" in checked_faces) and DeepFace.verify(frame, soulaymanImg.copy())['verified']:
            who = "soulayman"
            checked_faces.append(who)
        else:
            who = "no match"
    except Exception as e:
        print(e)
    return who


while True:
    ret, img = video_capture.read()
    if ret:
        #Since the operation == is faster than x%y==z
        if counter == 30:

            try:
                #The function that calls the deepface functions
                faces, names = detect_bounding_box_with_names(img)

                counter=1

            except Exception as e:
                print(e)
        counter += 1


        #The function that doesnt call the deepface functions but uses saved data from the last time we used detect_bounding_box_with_names
        detect_bounding_box(img)
          ######

        """ if face_match==True:
            cv2.putText(img, who,(20, 450),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0,255,0),3)
            print(who) """

        """if face_match == False:
            cv2.putText(img, "Nomatch", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 3)"""


        cv2.imshow("face detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()