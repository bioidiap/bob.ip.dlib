#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Test Units
"""
#==============================================================================
# Import what is needed here:
import numpy as np

from .. import FaceDetector

def test_face_detector():
    """
    Test FaceDetector class.
    """

    image = np.zeros((3, 100, 100))

    result = FaceDetector().detect_single_face(image)

    assert result is None

    image = np.ones((3, 100, 100))

    result = FaceDetector().detect_single_face(image)

    assert result is None