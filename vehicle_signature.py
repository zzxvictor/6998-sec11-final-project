from utils.data_loader import  DataLoader4Signature
from configs import constants
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_loader = DataLoader4Signature(constants.DATA_ROOT,
                                       constants.ANNOTATION_FILES)
    train, val = data_loader.load(batch_size=32, repeat=True)
    for idx, (img1, img2, labels) in enumerate(train.take(5)):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img1[1].numpy() + 0.5)
        axs[1].imshow(img2[1].numpy() + 0.5)
        plt.title(str(idx) + ':'+ str(labels[1]))
        plt.show()
