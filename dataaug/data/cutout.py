import torch
import numpy as np
from PIL import Image


class Cutout:
    """This is data augmentation via Cutout. https://arxiv.org/abs/1708.04552.

    Modified to run on PIL images.
    """

    def __init__(self, probability=1.0, mask_size=16, mask_color=(0.0, 0.0, 0.0)):
        """Cut-out with given area and mask color (in floats)"""
        super().__init__()
        self.p = probability
        self.mask_size = mask_size
        self.mask_color = torch.as_tensor(mask_color).mul(255).to(dtype=torch.uint8)[None, None, :]

        self._mask_size_half = mask_size // 2
        self._offset = 1 if mask_size % 2 == 0 else 0

    def __call__(self, pic):
        """run cutout."""
        if np.random.random() > self.p:
            return pic
        # handle PIL Image
        img = torch.as_tensor(np.array(pic, dtype=np.uint8, copy=True))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))

        # generate bounding box and fill
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(img.shape)
        img[bbx1:bbx2, bby1:bby2, :] = self.mask_color
        # return to PIL:
        pic = Image.fromarray(img.numpy(), mode="RGB")
        return pic

    def _rand_bbox(self, size):
        W = size[0]
        H = size[1]
        # uniform
        cx = np.random.randint(W + self._offset)
        cy = np.random.randint(H + self._offset)
        bbx1 = np.clip(cx - self._mask_size_half, 0, W)
        bby1 = np.clip(cy - self._mask_size_half, 0, H)
        bbx2 = np.clip(cx + self._mask_size_half, 0, W)
        bby2 = np.clip(cy + self._mask_size_half, 0, H)
        return bbx1, bby1, bbx2, bby2
