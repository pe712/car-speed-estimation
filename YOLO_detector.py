import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model: YOLO, classes=None, draw=False, treshold=0.5, margin=0.1):
        self.model = model
        self.classes = classes
        self.draw = draw
        self.threshold = treshold
        self.margin = margin

    def detect(self, frame, color=(255, 0, 0), thickness=4):
        detections = self.model.track(frame, persist=True)[0]
        detections_ = []
        for x1, y1, x2, y2, track_id, confidence_score, class_id in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, class_id = int(x1), int(
                y1), int(x2), int(y2), int(track_id), int(class_id)
            if not self.classes or class_id in self.classes:
                detections_.append(
                    [x1, y1, x2, y2, track_id, confidence_score])
                if self.draw and confidence_score > self.threshold:
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color=color, thickness=thickness
                    )
                    class_name = self.model.names[class_id]
                    cv2.putText(frame, f"{class_name}-{track_id}: {round(confidence_score, 2)}", (
                        x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)
        return {"frame": frame, "detections": detections_}

    @staticmethod
    def yolo2standard_bbox(x, y, w, h):
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
        return x1, y1, x2, y2

    def extract_best(self, frame, **kwargs):
        results = self.detect(frame, **kwargs)
        frame, detections = results["frame"], results["detections"]
        best_detection = max(detections, key=lambda x: x[-1])
        return {"frame": self.mask(frame, [best_detection])}

    def extract(self, frame, **kwargs):
        results = self.detect(frame, **kwargs)
        frame, detections = results["frame"], results["detections"]
        return {"frame": self.mask(frame, detections)}

    def mask(self, frame, detections):
        if not detections:
            return {"frame": frame}
        h, w = frame.shape[:2]
        mask = np.zeros_like(frame)
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            x1, y1, x2, y2 = int(x1-w*self.margin), int(y1-h*self.margin), int(x2+w*self.margin), int(y2+h*self.margin)
            mask[y1:y2, x1:x2] = 1
        return mask*frame
