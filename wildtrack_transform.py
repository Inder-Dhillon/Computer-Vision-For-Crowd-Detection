import intersecting_area as ia
import image_transform as imt
import annotations_viewer as av
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import MeanShift, estimate_bandwidth
import os
from os import listdir
from os.path import isfile, join
import pickle5
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import json

area_width = 3600
area_height = 1200

color = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255),(0,0,0)]
#Red, Green, Blue, Turqois, Purple,Yellow, Black


# For yolo------------------------------------------------------------------------------------------------
def get_detections_for_camera(d,num):
    alldet = [] # person number, camera view, x, y, confidence score
    cam = num
    # for all frames
    for frame, info in d.items():
        det = []
        person = 1
        for key in info:
            # print(key, info[key]['class'])
            if info[key]['class'] == 'person':
                # get midpoint of bottom of boundary box
                box = info[key]['xyxy']
                midpt = box[0] + round((box[2] - box[0]) / 2)  # xmin + (xmax-xmin/2)
                ymax = box[3]
                conf = info[key]['conf']
                det.append([person, cam, midpt, ymax, conf])
                person += 1
        alldet.append(np.array(det))
    return alldet


def load_annotations(_lst_files):
    """
    Decodes a JSON file & returns its content.
    """
    annotations = []
    for filename in _lst_files:
        if not os.path.exists(filename):
            raise FileNotFoundError("File %s not found." % filename)
        try:
            with open(filename, 'r') as _f:
                _data = json.load(_f)
            positions = []
            for location in _data:
                positionID = location["positionID"]
                X = -300 + 300 + 2.5 * (positionID % 480)
                Y = -900 + 90 + 2.5 * (positionID / 480)
                positions.append((int(X),int(Y)))
            annotations.append(positions)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode {filename}.")
        if not isinstance(_data, list):
            raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
        if len(_data) > 0 and not isinstance(_data[0], dict):
            raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
    return annotations

img_dir = r"00000000.png"
images = ia._load_images(ia.get_dir_paths("Wildtrack_dataset/Image_subsets"),_n=0, _ext='png')
rvec, tvec = ia.load_all_extrinsics(ia.get_dir_paths("Wildtrack_dataset/calibrations/extrinsic"))
cameraMatrices, distCoeffs = ia.load_all_intrinsics(ia.get_dir_paths("Wildtrack_dataset/calibrations/intrinsic_original"))
n_views = len(images)

outputs = []
homographyMatrices = []
points = []
for n in range(n_views):
        img = images[n]
        R = np.asarray(rvec[n])
        T = np.asarray(tvec[n])
        K = np.asarray(cameraMatrices[n])
        hmat, out = imt.birds_eye_view_from_camera_param(img, R, T, K)
        #locations = imt.inf.recognize_from_image(img, mode="pos")
        #new_locations = list(map(lambda p: imt.point_transform(p, np.linalg.inv(hmat)), locations))
        #for p in range(0,len(new_locations)):
        #        cv2.circle(out, new_locations[p], 20, color[n], -1)
        #        cv2.circle(blank, new_locations[p], 20, color[n], -1)
        outputs.append(out)
        homographyMatrices.append(hmat)
        #points = points + new_locations


# Calculate blended image
blended = outputs[0]
for i in range(len(outputs)):
    if i == 0:
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        blended = cv2.addWeighted(outputs[i], alpha, blended, beta, 0.0)
"""

"""

# Detections
path = './yolodetections'
#path = './det2'
pklfiles = [pf for pf in listdir(path) if isfile(join(path, pf))]

everycam = []
ct = 0
for f in pklfiles:
    with open(path + '/' + f, 'rb') as f:
        d = pickle5.load(f)
        c = get_centers(d,ct)
        everycam.append(c)
        ct +=1

print(len(everycam[0]))
#cam 1
a = everycam[0]
#frame 0
print(len(a[0]))

path = './annotations'
afiles = [pf for pf in listdir(path) if isfile(join(path, pf))]
print(afiles[0])
ann = load_annotations([afiles[386]])
print(len(ann[0]))




locations = []
# loop over all cam
for cam in everycam:
        frame0 = cam[386] #frame 0

        pos = []
        # get all person positions in that frame
        for l in frame0:
                cord = (int(l[2]),int(l[3]))
                pos.append(cord)
        locations.append(pos)

points = []
for n in range(n_views):
        pos = locations[n]
        hmat = homographyMatrices[n]
        new_locations = list(map(lambda p: imt.point_transform(p, np.linalg.inv(hmat)), pos))
        for p in range(0,len(new_locations)):
                #cv2.circle(outputs[n], new_locations[p], 20, color[n], -1)
                cv2.circle(blended, new_locations[p], 20, color[n], -1)
        points = points + new_locations


size = (300,900)
ott = cv2.resize(outputs[0],size)
cv2.imshow("pp", ott)


"""
for cam in everycam:
        pos = []
        for c in cam:
                frame = c 

                # get all person positions in that frame
                for l in frame:
                        cord = (int(l[2]),int(l[3]))
                        pos.append(cord)
        locations.append(pos)

print(len(locations))
print(len(locations[0]))

points = []
for n in range(n_views):
        pos = locations[n]
        hmat = homographyMatrices[n]
        new_locations = list(map(lambda p: imt.point_transform(p, np.linalg.inv(hmat)), pos))
        for p in range(0,len(new_locations)):
                cv2.circle(outputs[n], new_locations[p], 20, color[n], -1)
                cv2.circle(blended, new_locations[p], 20, color[n], -1)
        points = points + new_locations

"""
print(len(points))



"""
#plot locations on original image
for pos in locations[0]:
        cv2.circle(images[0], pos, 20, color[0], -1)

cv2.imshow("points", images[0])
"""


points = np.asarray(points)     #projected points
#points = np.asarray(ann[0])    //annotated points
print("p")
print(len(points))
print(len(np.asarray(ann[0])))
#points = points.T

#xmin, xmax = np.min(x), np.max(x) 
#ymin, ymax = np.min(y), np.max(y) 

# use grid search cross-validation to optimize the bandwidth
grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.0, 1, 20)},cv=20)
grid.fit(points)
print(grid.best_params_)

x = points[:,0]
y = points[:, 1]

xmin, xmax = 0, 1200
ymin, ymax = 0, 3500
# Peform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:100, ymin:ymax:100]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
print(xx.shape)

kernel = stats.gaussian_kde(values,bw_method=0.1)
f = np.reshape(kernel(positions).T, xx.shape)
print(points.shape)

"""
density = np.reshape(kernel(positions).T, (-1, 1))
density = np.hstack((positions.T, density))
print(density.shape)
#print(positions[:,0])
#print(kernel(positions[:,0]))
#print(f[0,0])


#Mean Shift Clustering to find Maxima (Person Centroids)
bandwidth = estimate_bandwidth(density, quantile=0.1, n_samples=500)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(density)
labels = ms.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)

from itertools import cycle
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
centers = []
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    #plt.plot(f[my_members, 0], f[my_members, 1], col + '.')
    plt.plot(density[my_members,0] , density[my_members,1], col + ',')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
    
    #print(cluster_center[1])
    #print(cluster_center[0])
    centers.append((cluster_center[0], cluster_center[1]))
    #print(cluster_center.shape)

print(centers)
plt.title('Estimated number of clusters: %d' % n_clusters_)

"""


"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(density[:,0], density[:,1], density[:,2], marker='o')
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='x', color='red', s=300, linewidth=5, zorder=10)
"""

# Plot Guassian 2D KDE
fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')      
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Annotations: Gaussian 2D KDE Frame 386 - BW=0.1')
#ax.invert_xaxis()
fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
ax.view_init(60, 35)


size = (960,540)
img = cv2.resize(images[0], size)
out = cv2.resize(outputs[0],size)
cv2.imshow("points", out)

b,g,r = cv2.split(outputs[0])       # get b,g,r
img = cv2.merge([r,g,b])     # switch it to rgb

fig = plt.figure()
plt.imshow(img)
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title('Projected Image')
plt.show()



cv2.waitKey()


