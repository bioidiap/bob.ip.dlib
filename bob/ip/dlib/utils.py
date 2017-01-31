import numpy
from bob.ip.facedetect import BoundingBox
import dlib

def bob_to_dlib_image_convertion(bob_image, change_color=True):
    """
    Bob stores color images as (C, W, H), where C is channels, W is the width and H is height; AND the order of the
    colors are R,G,B.
    On the other hand, Dlib do (W, H, C) AND the order of the colors are the same as OpenCV (B, G, R)
    """
    dlib_image = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]), dtype="uint8")
    if change_color:
        dlib_image[:, :, 0] = bob_image[2, :, :]  # B
        dlib_image[:, :, 1] = bob_image[1, :, :]  # G
        dlib_image[:, :, 2] = bob_image[0, :, :]  # R
    else:
        dlib_image[:, :, 0] = bob_image[0, :, :]  # B
        dlib_image[:, :, 1] = bob_image[1, :, :]  # G
        dlib_image[:, :, 2] = bob_image[1, :, :]  # R

    return dlib_image


def bounding_box_2_rectangle(bb):
    """
    Converrs a bob.ip.facedetect.BoundingBox to dlib.rectangle
    """

    assert isinstance(bb, BoundingBox)
    return dlib.rectangle(bb.topleft[1], bb.topleft[0],
                          bb.bottomright[1], bb.bottomright[0])




# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
