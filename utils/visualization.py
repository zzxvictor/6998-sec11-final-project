import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def drawBox(image, boxes):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    ax.set_axis_off()

    ax.imshow(image, aspect='auto')
    for i, box in enumerate(boxes):
        coor = [box[0], box[1]]
        coor[0] = np.clip(coor[0], a_min=0, a_max=image.shape[1])
        coor[1] = np.clip(coor[1], a_min=0, a_max=image.shape[0])
        w, h = box[2] - box[0], box[3] - box[1]
        w = np.clip(w, a_min=0, a_max=image.shape[1])
        h = np.clip(h, a_min=0, a_max=image.shape[0])

        rec = patches.Rectangle(coor, w, h, fill=False)
        ax.add_patch(rec)
    return fig