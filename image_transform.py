import cv2
import numpy as np
import inference as inf


def point_transform(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
            matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
            matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    return int(px), int(py)


def birds_eye_view_from_camera_param(img, rvec, tvec, camera_matrix, size=(1200, 3600)):
    rmat, _ = cv2.Rodrigues(rvec)
    rmat = np.delete(rmat, -1, axis=1)
    pose_matrix = cv2.hconcat([rmat, tvec])
    hmat = camera_matrix @ pose_matrix
    shift_origin = np.array([1, 0, -300, 0, 1, -90, 0, 0, 1]).reshape((3,3))
    hmat = hmat.dot(shift_origin)
    img = cv2.warpPerspective(img, hmat, size, flags=cv2.WARP_INVERSE_MAP)
    return cv2.transpose(img)


def translation_mat(dx, dy):
    return np.array([1, 0, dx, 0, 1, dy, 0, 0,     1]).reshape((3,3))


