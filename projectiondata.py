import json
import numpy as np
import intersecting_area as ia
import image_transform as imt
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import pickle5
# install torch from https://pytorch.org/get-started/locally/
from os import listdir
from os.path import isfile, join
import pickle



def birds_eye_view_from_camera_param(rvec, tvec, camera_matrix): # removed input img, size=(1200, 3600)
    rmat, _ = cv2.Rodrigues(rvec)
    rmat = np.delete(rmat, -1, axis=1)
    pose_matrix = cv2.hconcat([rmat, tvec])
    hmat = camera_matrix @ pose_matrix
    shift_origin = np.array([1, 0, -300, 0, 1, -90, 0, 0, 1]).reshape((3,3))
    hmat = hmat.dot(shift_origin)
    #img = cv2.warpPerspective(img, hmat, size, flags=cv2.WARP_INVERSE_MAP)
    #transpose = cv2.transpose(img)
    return hmat

def find_ground_plane_coor(pt, view): # removed input image_dir, camera
    #img = cv2.imread(image_dir)
    rvect = np.asarray(rvec[view])
    tvect = np.asarray(tvec[view])
    camMatrix = np.asarray(cameraMatrices[view])
    hmat = birds_eye_view_from_camera_param(rvect, tvect, camMatrix) # was out, hmat = ... before
    wptx, wpty = imt.point_transform(pt, np.linalg.inv(hmat))
    wp = [wptx, wpty]
    #invP = np.linalg.inv(hmat)
    #wp = invP.dot(pt)
    #wp = wp / wp[2]
    #print(camera, wp)
    return wp

def get_from_annotation(file):
    # get annotation information from json files
    with open(file) as f:
        data = json.load(f)

    #print(json.dumps(data, indent = 4, sort_keys=True))
    numppl = len(data)
    groundtruth = []  # columns: person ID, camera, x, y
    for num in range(0,len(data)):
        i = 0
        personID = data[num]['personID']
        for key in data:
            if i < 7:
                camera = data[num]['views'][i]['viewNum']
                xmax = data[num]['views'][i]['xmax']
                xmin = data[num]['views'][i]['xmin']
                midpt = xmin + round((xmax-xmin)/2)
                ymax = data[num]['views'][i]['ymax']
                groundtruth.append([personID, camera, midpt, ymax])
                i += 1
    groundtruth = np.array(groundtruth)
    #print(personID, camera, xmax, xmin, midpt, ymax)
    #print(groundtruth)
    return groundtruth, numppl

def gt_projection(mypath, file):
    groundtruth, numppl = get_from_annotation(mypath + '/' + file)
    groundtruthproj = []  # ground truth with projected points added
    for entry in groundtruth:
        if entry[2] != -1: # -1 means the person was missing from the frame
            pt = [entry[2], entry[3], 1]
            view = entry[1]
            wpt = find_ground_plane_coor(pt, view)
            wpt = [int(wpt[0]), int(wpt[1])]
            row = np.concatenate((entry, wpt))
            groundtruthproj.append(row)

    groundtruthproj = np.array(groundtruthproj)
    return groundtruthproj, numppl

def anno_Proj(file):
    # get 3D projections from annotated data (ground truth)
    with open(file) as f:
        data = json.load(f)

    numppl = len(data)
    annoproj = []  # columns: person ID, camera, x, y
    for num in range(0, len(data)):
        personID = data[num]['personID']  #grid 480 * 1440
        positionID = data[num]["positionID"]
        X = round(-300 + 300 + 2.5 * (positionID % 480))
        Y = round(-900 + 90 + 2.5 * (positionID / 480))
        #X = -3.0 + 0.025 * positionID % 480 #original
        #Y = -9.0+ 0.025 * positionID / 480
        annoproj.append([X, Y])
    annoproj = np.array(annoproj)
    return annoproj, numppl
# For yolo------------------------------------------------------------------------------------------------
def get_detections_for_camera(d,num):
    alldet = [] # person number, camera view, x, y, confidence score
    cam = num
    for image, info in d.items():
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

def frame_detections(everycam, framenum):
    # get detections in each frame from all camera views given frame number
    allframe = []
    for camview in range(len(everycam)):
        frame = everycam[camview][framenum]
        allframe.append(frame)
    frame_det = np.concatenate(allframe, axis=0)
    return frame_det

def frame_det_projections(detframe, threshold):
    # get ground plane projections for each frame (together with original 2D coordinates)
    detframeproj = []
    for entry in detframe:
        if entry[4] > threshold: # entry4 is confidence score
            pt = [entry[2], entry[3], 1]
            view = int(entry[1])
            wpt = find_ground_plane_coor(pt, view)
            if (0 <= wpt[0] <= 1200) and (0 <= wpt[1] <= 3600):

            #wpt = [int(wpt[0]), int(wpt[1])]
                row = np.concatenate((entry, wpt))
                detframeproj.append(row)

    detframeproj = np.array(detframeproj)
    return detframeproj


#----------------------------------------------------------------------
def GMM(datapts, numppl):
    #  Gaussian Mixture Model
    datapts = datapts.T #need for gmm
    gmm = GaussianMixture(n_components=numppl)
    gmm.fit(datapts)
    #predictions from gmm
    labels = gmm.predict(datapts)
    a_set = set(labels)
    number_of_unique_values = len(a_set)
    #print("Number of people: ", number_of_unique_values)
    #plt.scatter(datapts[:, 0], datapts[:, 1], s=3, c=labels, cmap='rainbow')


    # get "centroids" - samples with greatest probability
    centers = np.empty(shape=(gmm.n_components, datapts.shape[1]))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i], allow_singular=True).logpdf(datapts)
        centers[i, :] = datapts[np.argmax(density)]
    #plt.scatter(centers[:, 0], centers[:, 1], s=2)
    #plt.show()
    #print(centers.tolist())
    return centers

def save_dictionaries(dict):
    try:
        f1 = open('yolocenters', 'wb')
        pickle.dump(dict, f1)
        f1.close()
    except:
        print("Something went wrong")
    return
#-----------------------------------------------------------------------------------------

rvec, tvec = ia.load_all_extrinsics(ia.get_dir_paths("D:/Programming/CVProject/CVProject/Wildtrack_dataset/calibrations/calibrations/extrinsic"))
cameraMatrices, distCoeffs = ia.load_all_intrinsics(ia.get_dir_paths("D:/Programming/CVProject/CVProject/Wildtrack_dataset/calibrations/calibrations/intrinsic_zero"))

mypath = './annotations_positions'

# Get annotations for each frame
GT_projections = {}
GT_centers = {}
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#onlyfiles = ["00000000.json"]                                # CHANGE HERE
frame = 0
for file in onlyfiles:
    gtproj, numppl = anno_Proj(mypath + '/' + file)
    gtproj = gtproj.T
    gt_centers = GMM(gtproj, numppl)
    GT_centers["frame{:d}".format(frame)] = gt_centers
    GT_projections["frame{:d}".format(frame)] = gtproj
    GT_projections["numpplframe{:d}".format(frame)] = numppl
    frame += 1



# Detections
path = './yolodetections'
#path = './det2'
pklfiles = [pf for pf in listdir(path) if isfile(join(path, pf))]

everycam = [] # person number, camera view, x, y, confidence score
ct = 0
for f in pklfiles:
    with open(path + '/' + f, 'rb') as f:
        d = pickle5.load(f)
        c = get_detections_for_camera(d,ct)
        c = c
        everycam.append(c)
        ct +=1

yolo_proj = {}
yolo_centers = {}
yolo_subset = {}
yolosubsetall = {}
views_list = [0.,2.,4.,6.]
for frame in range(0,400):
    # get detections in each frame from all camera views
    detframe = frame_detections(everycam, frame)

    # get ground plane projections for each frame (together with original 2D coordinates)
    threshold = 0.2
    yoloproj = frame_det_projections(detframe, threshold) # person, cam, x, y, conf, X, Y
    yolo_proj["frame{:d}".format(frame)] = yoloproj
    yolosubset = []
    for row in yoloproj:
        if row[1] in views_list:
            yolosubset.append(row)
    yolosubset = np.array(yolosubset)
    yolosubsetall["frame{:d}".format(frame)] = yolosubset

    detectpts = np.array([yoloproj[:, 5], yoloproj[:, 6]])  # take projected points only
    detectptssub = np.array([yolosubset[:, 5], yolosubset[:, 6]])  # take projected points only

    #plt.scatter(detectpts[0,:], detectpts[1,:], s = 3, cmap='rainbow')
    #plt.show()
    ic = int((len(yoloproj) / 7))
    centers = GMM(detectpts,ic)
    #print(centers.tolist())
    yolo_centers["frame{:d}".format(frame)] = centers
    yolo_centers["avnumpplframe{:d}".format(frame)] = ic

    # for subset
    ic1 = int((len(yolosubset) / 4))
    centers_sub = GMM(detectptssub, ic1)
    yolo_subset["frame{:d}".format(frame)] = centers_sub
    yolo_subset["avnumpplframe{:d}".format(frame)] = ic1


save_dictionaries(GT_centers)
save_dictionaries(GT_projections)
save_dictionaries(yolo_centers)
save_dictionaries(yolo_proj)

plt.scatter(yolo_proj["frame0"][:,5], yolo_proj["frame0"][:,6], s = 3)
plt.scatter(GT_centers["frame0"][:,0], GT_centers["frame0"][:,1], s = 3, color='red')
plt.show()