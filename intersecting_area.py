#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Shows the intersecting area between the seven cameras of the WILDTRACK dataset,
which is the area considered for annotating the persons.
This script generates 3D points on the ground plane (z-axis is 0) using a grid
of size 1440x480, origin at (-300,  -90,    0) in centimeters (cm), and uses
step of 2.5 cm in both the axis. It then projects these points in each of the
views. Finally, it stores one frame per each view where the grid is shown.
The purpose of this code is to demonstrate how the provided calibration files
can be used, using the OpenCV library.

For information regarding the WILDTRACK dataset, see the following paper:
'WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection',
Tatjana Chavdarova, Pierre Baqué, Stéphane Bouquet, Andrii Maksai, Cijo Jose,
Timur Bagautdinov, Louis Lettry, Pascal Fua, Luc Van Gool, François Fleuret;
In Proceedings of The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2018, pp. 5030-5039. Available at: http://openaccess.thecvf.com/
content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html

To download the dataset visit: https://cvlab.epfl.ch/data/wildtrack

Copyright (c) 2008 Idiap Research Institute, http://www.idiap.ch/
Written by Tatjana Chavdarova <tatjana.chavdarova@idiap.ch>

This file is part of WILDTRACK toolkit.

WILDTRACK toolkit is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

WILDTRACK toolkit is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with WILDTRACK toolkit. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import cv2
from xml.dom import minidom
from os import listdir
from os.path import isfile, isdir, join
import xml.etree.ElementTree as ElementTree

# specific to the WILDTRACK dataset:
_grid_sizes = (1440, 480)
_grid_origin = (-300, -90, 0)
_grid_step = 2.5


def get_dir_paths(d):
    return [os.path.join(d, path) for path in os.listdir(d)]


def load_opencv_xml(filename, element_name, dtype='float32'):
    """
    Loads particular element from a given OpenCV XML file.
    """
    if not isfile(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        tree = ElementTree.parse(filename)
        rows = int(tree.find(element_name).find('rows').text)
        cols = int(tree.find(element_name).find('cols').text)
        return np.fromstring(tree.find(element_name).find('data').text,
                             dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(e)
        raise UnicodeDecodeError('Error while decoding file %s.' % filename)


def _load_images(_dirs, _n=0, _ext='png'):
    """
    Loads the _n-th image of each of the given directories.
    """
    _imgs = []
    for _, _dir in enumerate(_dirs):
        if not isdir(_dir):
            raise NotADirectoryError('%s is not a directory.' % _dir)

        files = [join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f)) and f.endswith(_ext)]
        if len(files) <= _n:
            raise IndexError("Found fewer files in %s than selected: %d" % (_dir, _n))
        _imgs.append(cv2.imread(sorted(files)[_n]))
    return _imgs


def load_all_extrinsics(_lst_files):
    """
    Loads all the extrinsic files, listed in _lst_files.
    """
    rvec, tvec = [], []
    for _file in _lst_files:
        xmldoc = minidom.parse(_file)
        rvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
        tvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])
    return rvec, tvec


def project_grid_points(_origin, _size, _offset, rvec, tvec, camera_matrices, dist_coef):
    """
    Generates 3D points on a grid & projects them into all the views,
    using the given extrinsic and intrinsic calibration parameters.
    """
    points = []
    for i in range(_size[0] * _size[1]):
        x = _origin[0] + _offset * (i % 480)
        y = _origin[1] + _offset * (i / 480)
        points.append(np.float32([[x, y, 0]]))  # ground points, z-axis is 0
    projected = []
    for c in range(len(camera_matrices)):
        imgpts, _ = cv2.projectPoints(np.asarray(points),  # 3D points
                                      np.asarray(rvec[c]),  # rotation rvec
                                      np.asarray(tvec[c]),  # translation tvec
                                      camera_matrices[c],  # camera matrix
                                      dist_coef[c])  # distortion coefficients
        projected.append(imgpts)
    return projected


def load_all_intrinsics(_lst_files):
    """
    Loads all the intrinsic files, listed in _lst_files.
    """
    _cameraMatrices, _distCoeffs = [], []
    for _file in _lst_files:
        _cameraMatrices.append(load_opencv_xml(_file, 'camera_matrix'))
        _distCoeffs.append(load_opencv_xml(_file, 'distortion_coefficients'))
    return _cameraMatrices, _distCoeffs


def draw_points(images, points):
    """
    Draws the 2D points in each of the images.
    The images are modified in-place.
    """
    if not isinstance(images, list):
        raise TypeError(f"Type mismatch. Found {type(images)}, expected list.")
    if not isinstance(points, list):
        raise TypeError(f"Type mismatch. Found {type(points)}, expected list.")
    if not len(images) == len(points):
        raise ValueError("Length mismatch: %d and %d" % (len(images), len(points)))
    for v in range(_n_views):
        for p in range(len(projected[v])):
            try:
                if (points[v][p].ravel())[0] >= 0 and (points[v][p].ravel())[1] >= 0:
                    cv2.circle(images[v], tuple(points[v][p].ravel()), 3, (255, 0, 0), -1)  # Blue
            except OverflowError:
                pass


if __name__ == '__main__':
    _folders = get_dir_paths("Wildtrack_dataset/Image_subsets")
    frames = _load_images(_folders, _n=0, _ext=".png")
    rvec, tvec = load_all_extrinsics(get_dir_paths("Wildtrack_dataset/calibrations/extrinsic"))
    cameraMatrices, distCoeffs = load_all_intrinsics(get_dir_paths("Wildtrack_dataset/calibrations/intrinsic_zero"))

    assert len(frames) == len(rvec) == len(tvec), "Inconsistent number of views"
    _n_views = len(frames)

    projected = project_grid_points(_grid_origin, _grid_sizes, _grid_step,
                                    rvec, tvec, cameraMatrices, distCoeffs)
    draw_points(frames, projected)

    for v in range(_n_views):
        cv2.imwrite("intersecting_area/grid" + str(v + 1) + '.png', frames[v])
    print(f"Images stored in intersecting_area/")
