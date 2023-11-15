import cv2


class VideoRunner:
    def __init__(self, video, callers=None, waitKey=30):
        if not callers:
            callers = []
        elif not isinstance(callers, list):
            callers = [callers]
        self.callers = callers
        self.video = video
        self.waitKey = waitKey

    def run(self):
        cap = cv2.VideoCapture(self.video)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            for caller in self.callers:
                frame = caller(frame)["frame"]
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(self.waitKey)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    VideoRunner("data/highway/static_front.mp4").run()