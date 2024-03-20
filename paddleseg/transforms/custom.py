
from albumentations.core.transforms_interface import DualTransform
import cv2
import numpy as np
import random

class RandomCopyMoveTransform(DualTransform):
    def __init__(
        self,
        max_h=0.8,
        max_w=0.8,
        min_h=0.05,
        min_w=0.05,
        mask_value=1,
        always_apply=False,
        p=0.5,
    ):
        """Apply cope-move manipulation to the image, and change the respective region on the mask to <mask_value>

        Args:
            max_h (float, optional): (0~1), max window height rate to the full height of image . Defaults to 0.5.
            max_w (float, optional): (0~1), max window width rate to the full width of image . Defaults to 0.5.
            min_h (float, optional): (0~1), min window height rate to the full height of image . Defaults to 0.05.
            min_w (float, optional): (0~1), min window width rate to the full width of image . Defaults to 0.05.
            mask_value (int, optional): the value apply the tampered region on the mask. Defaults to 255.
            always_apply (bool, optional): _description_. Defaults to False.
            p (float, optional): _description_. Defaults to 0.5.
        """
        super(RandomCopyMoveTransform, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value

    def _get_random_window(self, img_height, img_width, window_height=None, window_width=None):
        assert self.max_h < 1 and self.max_h > 0
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0

        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)

        if window_width == None or window_height == None:
            window_h = np.random.randint(l_min_h, l_max_h)
            window_w = np.random.randint(l_min_w, l_max_w)
        else:
            window_h = window_height
            window_w = window_width

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)

        return pos_h, pos_w, window_h, window_w

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.copy()
        H, W, _ = image.shape
        # copy region:
        c_pos_h, c_pos_w, c_window_h, c_window_w = self._get_random_window(H, W)

        # past region, window size is defined by copy region:
        self.p_pos_h, self.p_pos_w, self.p_window_h, self.p_window_w = self._get_random_window(
            H, W, c_window_h, c_window_w
        )

        copy_region = image[c_pos_h : c_pos_h + c_window_h, c_pos_w : c_pos_w + c_window_w, :]
        image[self.p_pos_h : self.p_pos_h + self.p_window_h, self.p_pos_w : self.p_pos_w + self.p_window_w, :] = (
            copy_region
        )
        return image

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """

        manipulated_region = np.full((self.p_window_h, self.p_window_w), 1)
        img = img.copy()
        img[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
        ] = self.mask_value
        return img


class RandomInpaintingTransform(DualTransform):
    def __init__(
        self,
        max_h=0.8,
        max_w=0.8,
        min_h=0.05,
        min_w=0.05,
        mask_value=1,
        always_apply=False,
        p=0.5,
    ):
        super(RandomInpaintingTransform, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value

    def _get_random_window(
        self,
        img_height,
        img_width,
    ):
        assert self.max_h < 1 and self.max_h > 0
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0

        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)

        window_h = np.random.randint(l_min_h, l_max_h)
        window_w = np.random.randint(l_min_w, l_max_w)

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)

        return pos_h, pos_w, window_h, window_w

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img = np.uint8(img)
        H, W, C = img.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        # inpainting region
        self.pos_h, self.pos_w, self.window_h, self.window_w = self._get_random_window(H, W)
        mask[
            self.pos_h : self.pos_h + self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = 1
        inpaint_flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS
        img = cv2.inpaint(img, mask, 3, inpaint_flag)
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
        img = img.copy()
        img[
            self.pos_h : self.pos_h + self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = self.mask_value
        return img
