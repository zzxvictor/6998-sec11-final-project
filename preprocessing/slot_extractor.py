import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
from shutil import copyfile
import numpy as np


class SlotExtractor:
    def __init__(self,
                 raw_img_root,
                 annotation_files,
                 output_dir,
                 scale=1.2):
        self.raw_img_root = raw_img_root
        self.annotation_files = annotation_files
        self.output_dir = output_dir
        self.ORIGINAL_W = 2595
        self.NEW_W = 1000
        self.ORIGINAL_H = 1944
        self.NEW_H = 750
        self.slot_coord = self._read_slot_coord()
        self.all_working_dir = [output_dir, output_dir + '/flattened',  output_dir + '/by_slot']
        self.scale = scale
        for directory in self.all_working_dir:
            if not os.path.isdir(directory):
                os.mkdir(directory)

    def flatten(self):
        for weather in os.listdir(self.raw_img_root):
            path = [self.raw_img_root, weather]
            for date in os.listdir('/'.join(path)):
                path.append(date)
                for camera_id in os.listdir('/'.join(path)):
                    path.append(camera_id)
                    for img in os.listdir('/'.join(path)):
                        path.append(img)
                        new_path = self.all_working_dir[1] + '/{}_{}_{}'.format(weather, camera_id, img)
                        copyfile('/'.join(path), new_path)
                        path.pop(-1)
                    path.pop(-1)
                path.pop(-1)


    def crop_by_slot(self):
        for img in os.listdir(self.all_working_dir[1]):
            weather, camera, date, time = img.strip('.jpg').split('_')
            camera = int(camera.strip('camera'))
            data = plt.imread(self.all_working_dir[1] + '/' + img)
            # in python image is y, x, channel
            for slot_id in self.slot_coord[camera]:
                print(camera, slot_id)
                bbox = self.slot_coord[camera][slot_id]
                y_min = int(np.clip(bbox[1] / self.scale, a_min=0, a_max=data.shape[0]))
                x_min = int(np.clip(bbox[0] / self.scale, a_min=0, a_max=data.shape[1]))
                y_max =  int(np.clip(bbox[3] * self.scale, a_min=0, a_max=data.shape[0]))
                x_max = int(np.clip(bbox[2] * self.scale, a_min=0, a_max=data.shape[1]))
                cropped = data[y_min: y_max, x_min: x_max]
                plt.imshow(cropped)
                plt.show()
                break
            break

    def _read_slot_coord(self):
        slot_coord = defaultdict(dict)
        for camera_file in self.annotation_files:
            camera_id = int(camera_file.strip('.csv')[-1])
            annotation = pd.read_csv(camera_file).to_numpy()
            for slot_info in annotation:
                slot_id = slot_info[0]

                top_left = [slot_info[1] * (self.NEW_W / self.ORIGINAL_W),
                               slot_info[2] * (self.NEW_H / self.ORIGINAL_H)]

                bottom_right = [(slot_info[1] + slot_info[3]) * (self.NEW_W / self.ORIGINAL_W),
                               (slot_info[2] + slot_info[4]) * (self.NEW_H / self.ORIGINAL_H)]

                slot_coord[camera_id][slot_id] = top_left + bottom_right
        return slot_coord