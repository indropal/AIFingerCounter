import cv2
import numpy as np
import HandTracker as handTrack
import time


def uprightFinger( img, fingerTipID, landMarkList ):
    """
        if detected Tip landmark is upright then this function will return 1 else 0
        Handle the Thumb separately --> as the thumb is oriented sideways on the hand ...By checking for Left or Right hand...

    """
    detect = 0;
    hand = "Left" if (landMarkList[0]['X'] > landMarkList[1]['X']) else "Right"

    cv2.putText( img, str(landMarkList[fingerTipID]['X'])+' , '+str(landMarkList[fingerTipID]['Y'])  , (landMarkList[fingerTipID]['X'], landMarkList[fingerTipID]['Y']), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3 )
    cv2.putText( img, hand, (landMarkList[0]['X'], landMarkList[0]['Y']), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 20, 100), 2 )

    if( fingerTipID == 4 ):
        """
            if Left hand, the the stem of the thum (ID: 1) will be on the left side of the wrist (ID: 0) & vice vaersa for the Right
            when we close or palm, the left thumb goes to the other extreme end of the palm & similarly for the right hand  -- Exploit this property for the thumb
        """
        if (landMarkList[0]['X'] > landMarkList[2]['X']):
            #print('LeftHand') # Detected the Left Hand
            if( landMarkList[4]['X'] > landMarkList[3]['X']):# or landMarkList[4]['X'] > landMarkList[2]['X'] ):
                #print("The Left Hand is closed") # Left Thumb is closed
                detect = 0;
            else: 
                detect = 1;
        else:
            #print('RightHand') # Detected teh Right Hand
            if( landMarkList[4]['X'] < landMarkList[3]['X']):# or landMarkList[4]['X'] < landMarkList[2]['X'] ):
                #print("The Right Hand is closed") # Right Thumb is closed
                detect = 0;
            else:
                detect = 1;
        
        return img, detect

    if ( (landMarkList[fingerTipID]['Y'] > landMarkList[fingerTipID - 2]['Y']) or (landMarkList[fingerTipID]['Y'] > landMarkList[fingerTipID - 3]['Y'])):
        detect = 0
    elif ( (landMarkList[fingerTipID]['Y'] < landMarkList[fingerTipID - 2]['Y']) or (landMarkList[fingerTipID]['Y'] < landMarkList[fingerTipID - 3]['Y']) ):
        detect = 1
        
    return img, detect;


def main():
  
    cap = cv2.VideoCapture(0);

    widthCam, heightCam = 720, 480;

    # set width & height if the image from camera
    cap.set(3, widthCam) # set width
    cap.set(4, heightCam) # set height

    prevTime = 0

    # Creating the Hand Detector object
    detector = handTrack.fingerTip( max_num_hands = 1, min_detection_confidence = 0.6 )

    fingerTipIDs = [4, 8, 12, 16, 20] # 'landmark_ID' --> tip of | Index: 8 | Thumb: 4 | Middle: 12 | Third: 16 | Pinky: 20

    while True:
        success, img = cap.read() # read the frame from camera feed

        # retrieve the Hand landmarks
        img = detector.findHands(img)
        landmarkList = detector.specificHandPosition( img, draw = False )


        # retrieve the coordinates for each of the Tips of the finger ...
        uprightFingerCounter = [] # keep count of the number of fingers opened
        count = 0; # number of fingers displayed

        if ( len(landmarkList) != 0 ):
            
            for lndMrk in landmarkList:
                if( lndMrk['landmark_ID'] in fingerTipIDs ):
                    img, detection = uprightFinger( img, lndMrk['landmark_ID'], landmarkList )
                    uprightFingerCounter.append( detection )

        if(len(uprightFingerCounter) > 0):
            count = sum(uprightFingerCounter)

        if ( len(landmarkList) != 0 ):
            print(uprightFingerCounter)
            cv2.rectangle( img, (25, 80), (55, 135), (150, 50, 100), cv2.FILLED )
            cv2.putText( img, str(count), (30, 120 ), cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 255), 2 )


        cv2.imshow("Image", img) # show the image obtained from the camera feed
        cv2.waitKey(1)


if __name__ == "__main__":
    main()