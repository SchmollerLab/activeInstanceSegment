import math
import matplotlib.pyplot as plt
from skimage import exposure, io


def show_image(ims, normalize=True):
    if not isinstance(ims, list):
        if normalize:
            im_cont = exposure.equalize_adapthist(ims)
        else:
            im_cont = ims
        plt.imshow(im_cont)
    
    else:
        fig = plt.figure(figsize=(16, 10))

        num_figures = len(ims)
        cols = 5
        rows = int(math.ceil(num_figures/cols))

        for i in range(num_figures):
            if normalize:
                im_cont = exposure.equalize_adapthist(ims[i])
            else:
                im_cont = ims[i]
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(im_cont)
    
    plt.show()