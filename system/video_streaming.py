import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


class VideoReader:
    def __init__(self,
                 video_path,
                 annotation_path,
                 fps=30,
                 delimiter=' '):
        self.video_path = video_path
        if os.path.isfile(annotation_path):
            self.annotation = self._read_annotation(annotation_path, delimiter)
        else:
            self.annotation = None
        self.fps = fps

    def start(self):
        frames = self._read_video(self.video_path, self.annotation)
        for counter, img in enumerate(frames):
            if self.annotation is None:
                print('first frame produced. Please use it for annotation!')
                return
            if counter % self.fps == 0:
                self._display_slot_location(img, self.annotation, counter / self.fps)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            if counter % (5 * self.fps) == 0:
                img_dict = self._extra_slot(img, self.annotation)
                yield img_dict, counter / self.fps

    @classmethod
    def _extra_slot(cls, img, annotation):
        imgs_dict = {}
        for slot_id in annotation.keys():
            x, y, w, h = cls._scale_bounding_box(annotation[slot_id], img)
            slot_img = img[y: y + h, x: x + w ]
            imgs_dict[slot_id] = slot_img
        return imgs_dict

    @classmethod
    def _read_video(cls, path, annotation):
        video = cv2.VideoCapture(str(path))
        counter = 0
        while video.isOpened():
            ok, img = video.read()
            if not ok:
                break
            if counter == 0:
                cv2.imwrite("first_frame.jpg", img)
            yield img
            counter += 1
        video.release()

    @classmethod
    def _read_annotation(cls, path, delim=' '):
        slot_location = {}
        fp = open(path, 'r')
        lines = fp.readlines()
        for line in lines:
            slot, x, y, w, h = line.split(delim)
            slot_location[slot] = list(map(float, [x, y, w, h]))
        fp.close()
        return slot_location

    @classmethod
    def _display_slot_location(cls, img, annotation, secs):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 10)
        ax.imshow(img)
        plt.title('{:.2f} secs'.format(secs))
        for slot in annotation.keys():
            x, y, w, h = cls._scale_bounding_box(annotation[slot], img)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 10, 'Slot {}'.format(slot), c='r')
        ax.axis('off')
        ax.margins(0)
        # convert canvas to image
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        cv2.imshow('cam feed', img)
        plt.close(fig)

    @classmethod
    def _scale_bounding_box(cls, coord, img):
        height, width, _ = img.shape
        x, y = coord[0] * width, coord[1] * height
        w, h = coord[2] * width, coord[3] * height
        coord = np.array([(x - w / 2), y - h / 2, w, h])
        return np.round(coord).astype(np.int32).tolist()