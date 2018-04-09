#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Fri 17 Jun 2016 10:41:36 CEST

import numpy
import os
import bob.core
import bob.io.image
logger = bob.core.log.setup("bob.ip.dlib")
bob.core.log.set_verbosity_level(logger, 3)
import dlib
from .utils import bounding_box_2_rectangle
from .FaceDetector import FaceDetector


def download_file(url, out_file):
    """Downloads a file from a given url

    Parameters
    ----------
    url : str
        The url to download form.
    out_file : str
        Where to save the file.
    """
    from bob.io.base import create_directories_safe
    import os
    create_directories_safe(os.path.dirname(out_file))

    import sys
    if sys.version_info[0] < 3:
        # python2 technique for downloading a file
        from urllib2 import urlopen
        with open(out_file, 'wb') as f:
            response = urlopen(url)
            f.write(response.read())

    else:
        # python3 technique for downloading a file
        from urllib.request import urlopen
        from shutil import copyfileobj
        with urlopen(url) as response:
            with open(out_file, 'wb') as f:
                copyfileobj(response, f)

class DlibLandmarkExtraction(object):
    """
    Binds to the DLib landmark detection using the shape_predictor_68_face_landmarks,

    This facial landmark detector is an implementation of [Kazemi2014]_

    Parameters
    ----------

    model: :py:class:`str`
      Path to the dlib pretrained model, if **None**, the model will be downloaded.

    bob_landmark_format: :py:class:`bool`
      If **True**, `__call__` will return the landmarks with Bob dictionary keys ('leye', 'reye', `nose`, .....).
      If **False**, `__call__` will return a list with the detected landmarks

    """

    def __init__(self, model=None, bob_landmark_format=False):

        default_model = os.path.join(DlibLandmarkExtraction.get_modelpath(), "shape_predictor_68_face_landmarks.dat")
        if model is None:
            self.model = default_model
            if not os.path.exists(self.model):
                DlibLandmarkExtraction.download_dlib_model()
        else:
            self.model = model
            if not os.path.exists(self.model):
                raise ValueError("Model not found: {0}".format(self.model))

        self.face_detector = FaceDetector()
        self.predictor = dlib.shape_predictor(self.model)
        self.bob_landmark_format = bob_landmark_format


    def __call__(self, image, bb=None, xy_output=False):

        # Detecting the face if the bounding box is not passed
        if bb is None:
            bb = self.face_detector.detect_single_face(image)

            if bb is None:
                return None

            bb = bounding_box_2_rectangle(bb[0])
        else:
            bb = bounding_box_2_rectangle(bb)

        if bb is None:
            raise ValueError("Face not found in the image.")

        points = self.predictor(bob.io.image.to_matplotlib(image), bb)
        if self.bob_landmark_format:
            points = list(map(lambda p: (p.y, p.x), points.parts()))
            bob_landmarks = dict()

            bob_landmarks['leye'] = ((points[37][0] + points[40][0])//2,
                                     (points[37][1] + points[40][1])//2)

            bob_landmarks['reye'] = ((points[43][0] + points[46][0])//2,
                                     (points[43][1] + points[46][1])//2)

            bob_landmarks['nose'] = (points[33][0], points[33][1])
            bob_landmarks['mouthleft'] = (points[49][0], points[49][1])
            bob_landmarks['mouthright'] = (points[55][0], points[55][1])

            return bob_landmarks

        else:
            if xy_output:
                return list(map(lambda p: (p.x, p.y), points.parts()))
            else:
                return list(map(lambda p: (p.y, p.x), points.parts()))

    @staticmethod
    def get_modelpath():
        import pkg_resources
        return pkg_resources.resource_filename(__name__, 'data')

    #url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

    @staticmethod
    def download_dlib_model():
        """
        Download and extract the dlib model face detection model from
        """

        """
        Download and extract the DLIB files
        """
        import zipfile
        zip_file = os.path.join(DlibLandmarkExtraction.get_modelpath(),
                                "shape_predictor_68_face_landmarks.dat.bz2")
        urls = [
            # This is a private link at Idiap to save bandwidth.
            "http://beatubulatest.lab.idiap.ch/private/wheels/gitlab/"
            "shape_predictor_68_face_landmarks.dat.bz2",
            # this works for everybody
            "http://dlib.net/files/"
            "shape_predictor_68_face_landmarks.dat.bz2",
        ]

        for url in urls:
            try:
                logger.info(
                    "Downloading DLIB model from "
                    "{} ...".format(url))
                download_file(url, zip_file)
                break
            except Exception:
                logger.warning(
                    "Could not download from the %s url", url, exc_info=True)
        else:  # else is for the for loop
            if not os.path.isfile(zip_file):
                raise RuntimeError("Could not download the zip file.")

        # Unzip
        logger.info("Unziping in {0}".format(DlibLandmarkExtraction.get_modelpath()))
        
        t = bz2.BZ2File(zip_file)
        open(os.path.join(DlibLandmarkExtraction.get_modelpath(),'shape_predictor_68_face_landmarks.dat'), 'wb').write(t.read())
        t.close()        
        
        os.unlink(zip_file)
        

