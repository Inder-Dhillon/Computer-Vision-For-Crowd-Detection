import intersecting_area as ia
import image_transform as imt
import cv2
import numpy as np

area_width = 3600
area_height = 1200

img_dir = r"00000000.png"
images = ia._load_images(ia.get_dir_paths("Wildtrack_dataset/Image_subsets"))
rvec, tvec = ia.load_all_extrinsics(ia.get_dir_paths("Wildtrack_dataset/calibrations/extrinsic"))
cameraMatrices, distCoeffs = ia.load_all_intrinsics(ia.get_dir_paths("Wildtrack_dataset/calibrations/intrinsic_original"))
rvec = np.asarray(rvec[0])
tvec = np.asarray(tvec[0])
cameraMatrix = np.asarray(cameraMatrices[0])
distCoeffs = np.asarray(distCoeffs[0])

img = cv2.imread(img_dir)
out = imt.birds_eye_view_from_camera_param(img, rvec, tvec, cameraMatrix)

cv2.imshow("output.jpg", out)
cv2.waitKey()
