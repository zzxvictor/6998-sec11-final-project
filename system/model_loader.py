from vehicle_detection import detector
import numpy as np


class LoadModelTF:
    @classmethod
    def load_model(cls, model_path):
        model = detector.get_detector()
        model.predict(np.random.randn(1, 150, 150, 3))
        model.load_weights(model_path)
        return model