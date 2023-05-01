import cv2
import mediapipe as mp

class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, minDetectionCon=10):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img)
        # print(self.results.detections)
        boxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boxC = detection.location_data.relative_bounding_box
                print(boxC)
                height, width, c = img.shape
                box = int(boxC.xmin * width), int(boxC.ymin * height - 100), \
                       int(boxC.width * width), int(boxC.height * height + 100)
                cx, cy = box[0] + (box[2] // 2), \
                         box[1] + (box[3] // 2)
                bboxInfo = {"id": id+1, "bbox": box, "score": detection.score, "center": (cx, cy)}
                # print(bboxInfo)
                boxs.append(bboxInfo)
                if draw:
                    img = cv2.rectangle(img, box, (255, 0, 255), 2)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (box[0], box[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, boxs


def main():
    cap = cv2.VideoCapture(0)
    
    ws, hs = 1280, 720
    cap.set(3, ws)
    cap.set(4, hs)

    if not cap.isOpened():
        print("Can not open the Camera")
        exit()
        
    detector = FaceDetector()
    while True:
        ret, img = cap.read()
        img1, bboxs = detector.findFaces(img)

        if bboxs:
            fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
            pos = [fx, fy]
            # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
            cv2.circle(img1, center, 5, (255, 0, 255), cv2.FILLED)
            
            cv2.circle(img1, (fx, fy), 80, (0, 0, 255), 2)
            cv2.putText(img1, str(pos), (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2 )
            cv2.line(img1, (0, fy), (ws, fy), (0, 0, 0), 2)  # x line
            cv2.line(img1, (fx, hs), (fx, 0), (0, 0, 0), 2)  # y line
            cv2.circle(img1, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
            cv2.putText(img1, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3 )
        
        else:
            cv2.putText(img1, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            cv2.circle(img1, (640, 360), 80, (0, 0, 255), 2)
            cv2.circle(img1, (640, 360), 15, (0, 0, 255), cv2.FILLED)
            cv2.line(img1, (0, 360), (ws, 360), (0, 0, 0), 2)  # x line
            cv2.line(img1, (640, hs), (640, 0), (0, 0, 0), 2)  # y line

        cv2.imshow("Image", img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()