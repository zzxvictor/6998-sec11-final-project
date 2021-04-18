import torch
#from vehicle_detection import detector
#import numpy as np

# class LoadModelTF:
#     @classmethod
#     def load_model(cls, model_path):
#         model = detector.get_detector()
#         model.predict(np.random.randn(1, 150, 150, 3))
#         model.load_weights(model_path)
#         return model


class LoadModelTorch:
    @classmethod
    def load_model(cls, class_callable, model_path, use_gpu=False):
        model = class_callable()
        if use_gpu:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path,
                                             map_location=torch.device('cpu'))
                                  )
        model.eval()
        return model
