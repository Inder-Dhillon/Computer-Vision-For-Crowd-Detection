import cv2
import numpy as np
import inference as inf


def compute_perspective_transform(corner_points, image, width=270, height=480):
    """ Compute the transformation matrix
    @ corner_points : 4 corner points selected from the image
    @ height, width : size of the image
    return : transformation matrix and the transformed image
    """
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def point_transform(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
            matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
            matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    return int(px), int(py)


img_dir = "test_photo3.jpg"
img = cv2.imread(img_dir)
corner_points = np.array([[0, 402], [484, 297], [600, 753], [1180, 477]], dtype="float32")
mat, wrp_img = compute_perspective_transform(corner_points, img)
# print(locations)

locations = inf.recognize_from_image("test_photo3.jpg", mode="pos")
new_locations = list(map(lambda p: point_transform(p, mat), locations))
for position in new_locations:
    cv2.circle(wrp_img, position, 3, (0,255,0), 3)

cv2.imshow("output", wrp_img)
cv2.imshow("input", img)
cv2.waitKey(0)
