
import cv2


class YoloDetector:
    def __init__(self, model, classes=None, draw=False, treshold=0.5):
        self.model = model
        self.classes = classes
        self.draw = draw
        self.threshold = treshold

    def detect(self, frame, color=(255, 0, 0), thickness=4):
        detections = self.model.predict(frame)[0]
        detections_ = []
        for x1, y1, x2, y2, confidence_score, class_id in detections.boxes.data.tolist():
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
            if not self.classes or class_id in self.classes:
                detections_.append([x1, y1, x2, y2, confidence_score])
                if self.draw and confidence_score > self.threshold:
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color=color, thickness=thickness
                    )
                    class_name = self.model.names[class_id]
                    cv2.putText(frame, f"{class_name}: {round(confidence_score, 2)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)
        return {"frame": frame, "detections": detections_}

    @staticmethod
    def yolo2standard_bbox(x, y, w, h):
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
        return x1, y1, x2, y2
