import numpy as np
import cv2
from keras.models import load_model


class ColorClassifier:
    
    def __init__(self, detector: callable, color_classification_model, labels):
        self.detector = detector
        self.color_classification_model = load_model(color_classification_model)
        self.labels = labels

    def predict(self, frame):
        results = self.detector(frame)
        frame, detections = results["frame"], results["detections"]
        _detections = []
        np.random.shuffle(detections)
        detections = detections[:1]
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            img = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
            img = np.expand_dims(img, axis=0)
            color_scores = self.color_classification_model.predict(img, batch_size=1)
            color = self.labels[np.argmax(color_scores)]
            print(color_scores)
            cv2.putText(frame, color, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            detection.append(color_scores)
            _detections.append(detection)
            mask = np.zeros_like(frame)
            mask[y1:y2, x1:x2] = 1
            frame = frame * mask
        return {"frame": frame, "detections": _detections}


