import cv2
import mediapipe as mp
import time



class fingerTip():

    def __init__( self, static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5 ):
        self.static_image_mode  = static_image_mode;
        self.max_num_hands = max_num_hands;
        self.min_detection_confidence = min_detection_confidence;
        self.min_tracking_confidence = min_tracking_confidence;

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands( self.static_image_mode,
                                         self.max_num_hands,
                                         self.min_detection_confidence,
                                         self.min_tracking_confidence
                                        );

        """ 
            '.Hands( static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence )' ::

            ---> 
                "static_image_mode" --> parameter for Detection Or Tracking -- if set to False, then based on the confidence level it will track the detected Hand
                "max_num_hands" --> maximum Number of Hands to detect & track
                "min_detection_confidence" --> min. model confidence threshold for Hand Detection
                "min_tracking_confidence" --> min. model confidence threshold for Hand Tracking -- if confidence value goes below the set threshold, then the model will initiate Detection
        """

        self.mpDraw = mp.solutions.drawing_utils; # the drawing utilities function for drawing lines and dots of the tracked/detected hand

    def findHands(self, img):

        """
            Method to detect/track Hands using MediaPipe & mark out finger tips from the 'img' - image variable
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); # convert captured image to RGB colors
        results = self.hands.process(imgRGB) # process the captured RGB image from the webcam using Mediapipe method
        # "results.multi_hand_landmarks" --> contains results for Hand Detection / Tracking
        if(results.multi_hand_landmarks):
            # if 'Hand' is being detected & tracked then, process them.
            for handLandmarks in results.multi_hand_landmarks:
                #print(handLandmarks); # print out the detected hand landmark features.

                # obtain the details for each individual Hand Landmarks -- Each landmark (a total of 21 such Landmarks are detected) along with coordinates
                for id, landMark in enumerate(handLandmarks.landmark):
                    # print(id, landMark) # Usually the landMark coordinates are in Ratio i.e. 0-1 values so that they can be scaled accordingly with changing image size
                    height, width, channel = img.shape
                    imgX, imgY = int(landMark.x * width), int(landMark.y * height)
                    
                    # targetting specific landmarks with their ID values -- tip of every finger & the wrist
                    if id == 0: # wrist 
                        # if 'id' --> 0 landmark is detected, then color it on the image feed differently
                        cv2.circle(img, (imgX, imgY), 15, (234, 234, 101), cv2.FILLED )
                        # Landmark with 'id = 0' will have detected landmark covered with a circle with RGB(234, 234, 101)

                    if id == 4: # thumb tip
                        # if 'id' --> 4 landmark is detected, then color it on the image feed differently
                        cv2.circle(img, (imgX, imgY), 15, (255, 128, 128), cv2.FILLED )
                        # Landmark with 'id = 4' will have detected landmark covered with a circle with RGB(255, 128, 128)

                    if id == 8: # index finger tip
                        # if 'id' --> 8 landmark is detected, then color it on the image feed differently                
                        cv2.circle(img, (imgX, imgY), 15, (255, 128, 255), cv2.FILLED )

                    if id == 12: # middle finger tip
                        # if 'id' --> 12 landmark is detected, then color it on the image feed differently                
                        cv2.circle(img, (imgX, imgY), 15, (150, 150, 255), cv2.FILLED )

                    if id == 16: # third finger tip
                        # if 'id' --> 16 landmark is detected, then color it on the image feed differently                
                        cv2.circle(img, (imgX, imgY), 15, (0, 127, 255), cv2.FILLED )

                    if id == 20: # pinky finger tip
                        # if 'id' --> 16 landmark is detected, then color it on the image feed differently                
                        cv2.circle(img, (imgX, imgY), 15, (200, 255, 75), cv2.FILLED )

                self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS); # draw the detected hand's landmark features on the image feed using mediapipe's utility functions & 'mpHands.HAND_CONNECTIONS' draws the connecting lines between the features
            

                
        return img; # return the image on which detection/tracking has been performed

    def specificHandPosition(self, img, handIdx = 0, draw = True ):

        """
            method to return the Landmarks for a particular Hand being detected 
            Specific Hand is denoted by the 'handIdx'
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); # convert captured image to RGB colors
        results = self.hands.process(imgRGB) # process the captured RGB image from the webcam using Mediapipe method
        landMarkList = []
        # "results.multi_hand_landmarks" --> contains results for Hand Detection / Tracking

        if ( results.multi_hand_landmarks ):
            myHand = results.multi_hand_landmarks[handIdx]

            for idx, landmark in enumerate( myHand.landmark ):
                # retireve required info
                height, width, channel = img.shape;
                imgX, imgY = int(landmark.x * width), int(landmark.y * height)
                landMarkList.append( { 'landmark_ID': idx, 'X': imgX, 'Y': imgY } )

                if (draw):
                    cv2.circle(img, (imgX, imgY), 7, (0, 128, 255), cv2.FILLED); # draw the detected hand's landmark features on the image feed using mediapipe's utility functions & 'mpHands.HAND_CONNECTIONS' draws the connecting lines between the features

        return landMarkList

                
def main():
    prevT = 0;
    currT = 0;
    cap = cv2.VideoCapture(0); # '1' passed to get video feed from webcam

    # create new fingerTip class object.
    handDetector = fingerTip()

    while True:
        success, img = cap.read()

        #img = handDetector.findHands(img);
        landMarkCoordinates = handDetector.specificHandPosition( img, 0, draw = True )  # returns list of dictionary attributes --> { 'landmark_ID': idx, 'X': imgX, 'Y': imgY }

        if ( len(landMarkCoordinates) != 0):
            for i in landMarkCoordinates:
                if i['landmark_ID'] == 0:
                    print(i)

        currT = time.time()
        framesPerSecond = 1 / (currT - prevT)
        prevT = time.time()
        cv2.putText(img, str(int(framesPerSecond)), (20, 50) , cv2.FONT_ITALIC, 1.5, (255,255,128), 3 );

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()