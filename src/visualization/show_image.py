import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from skimage import exposure, io


def show_image(ims, normalize=True):
    # figure(figsize=(10, 10), dpi=80)
    if not isinstance(ims, list):
        if normalize:
            im_cont = exposure.equalize_adapthist(ims)
        else:
            im_cont = ims
        plt.imshow(im_cont)

    else:

        fig = plt.figure(figsize=(22, 15))

        num_figures = len(ims)
        cols = 3
        rows = int(math.ceil(num_figures / cols))

        for i in range(num_figures):
            if normalize:
                im_cont = exposure.equalize_adapthist(ims[i])
            else:
                im_cont = ims[i]
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(im_cont)

    plt.show()
