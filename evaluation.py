import numpy as np
import munkres
from munkres import Munkres
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy
import scipy.stats as st
import pandas as pd

def hungarian_matching(GT_coordinates, det_coordinates, radius_match, verbose=False):
    #radius_match = 50 cm -> about the width of human body
    # reference: https://github.com/pierrebaque/DeepOcclusion
    n_dets = det_coordinates.shape[0]
    n_gts = GT_coordinates.shape[0]

    n_max = max(n_dets, n_gts)

    matrix = np.zeros((n_max, n_max)) + 1

    TP_det, TP_gt = [], []
    for i_d in range(n_dets):
        for i_gt in range(n_gts):
            if ((det_coordinates[i_d, 0] - GT_coordinates[i_gt, 0]) ** 2 + (
                    det_coordinates[i_d, 1] - GT_coordinates[i_gt, 1]) ** 2) <= radius_match ** 2:
                matrix[i_d, i_gt] = 0
                # get paired detected and ground truth
                TP_det.append((int(det_coordinates[i_d, 0]), int(det_coordinates[i_d, 1])))
                TP_gt.append((int(GT_coordinates[i_gt, 0]), int(GT_coordinates[i_gt, 1])))

    m = Munkres()
    indexes = m.compute(copy.copy(matrix))


    total = 0
    TP = []  # True positive
    FP = []  # False positive
    FN = []  # False negative
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        if verbose:
            print
            '(%d, %d) -> %d' % (row, column, value)
        if value == 0:
            TP.append((int(det_coordinates[row, 0]), int(det_coordinates[row, 1])))

        if value > 0:
            if row < n_dets:
                FP.append((int(det_coordinates[row, 0]), int(det_coordinates[row, 1])))
            if column < n_gts:
                FN.append((int(GT_coordinates[column, 0]), int(GT_coordinates[column, 1])))
    if verbose:
        print
        'total cost: %d' % total

    return total, np.asarray(TP), np.asarray(FP), np.asarray(FN), np.asarray(TP_det), np.asarray(TP_gt)

def MODA(mt, fp, groundtruth):
    # Multiple Object Detection Accuracy (MODA) - for frame
    # mt is missed detection count = FALSE NEGATIVE
    # fp is false positive count
    # cm & cf are cost functions for missed detects and false positives
    # Ng is the number of ground truth objects in the frame
    Ng = len(groundtruth)
    cm = 1
    cf = 1
    MODA = 1 - (((cm*mt) + (cf*fp)) / Ng)
    return MODA

def NMODA(mtall, fpall, gtall):
    # Multiple Object Detection Accuracy (MODA) - for entire sequence
    # mt is missed detection count  = FALSE NEGATIVE
    # fp is false positive count
    # cm & cf are cost functions for missed detects and false positives
    # Ng is the number of ground truth objects in the frame
    cm = 1
    cf = 1
    num_frames = len(gtall)
    numerator = []
    #denominator = []
    for f in range(0, num_frames):
        n = (cm * mtall[f]) + (cf * fpall[f])
        numerator.append(n)
        #Ng = len(gtall[f])
        #denominator.append(Ng)

    NMODA = 1 - (np.sum(numerator)/np.sum(gtall))
    return NMODA

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction
    # + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def groundplane_box(centers):
# assume person occupies 0.5 x 0.5 m area on ground plane. 1 pixel = 0.01 m, so 0.5 radius = 50 pixels
    addr = 50
    boxes = []
    for coor in centers:
        x1 = coor[0] - addr
        x2 = coor[0] + addr
        y1 = coor[1] - addr
        y2 = coor[1] + addr
        box = [x1, x2, y1, y2]
        boxes.append(box)
    return boxes

def MODP(mapped_detect, mapped_groundtruth):
    # Multiple Object Detection Precision - for frame
    # mapped overlap ratio mAR = sum(intersection over union)
    # Nm = number of mapped object pairs in frame
    mapped_detectb = groundplane_box(mapped_detect)
    mapped_groundtruthb = groundplane_box(mapped_groundtruth)
    Nm = len(mapped_detect)
    iou = []
    for i in range(len(mapped_detect)):
        iou.append(intersection_over_union(mapped_detectb[i], mapped_groundtruthb[i]))

    mAR = np.sum(iou)
    MODP = mAR / Nm
    return MODP

def NMODP(detect_all, groundtruth_all):
    # Multiple Object Detection Precision - for entire sequence
    num_frames = len(groundtruth_all)
    store = []
    for f in range(0, num_frames):
        s = MODP(detect_all[f], groundtruth_all[f])
        store.append(s)

    NMODP = (np.sum(store)) / num_frames
    return NMODP

def precision(true_positive, false_positive):
    p = true_positive / (true_positive + false_positive)
    return p

def recall(true_positive, false_negative):
    r = true_positive / (true_positive + false_negative)
    return r

def GMM(datapts, numppl): # GMM with plots
    datapts = datapts.T
    gmm = GaussianMixture(n_components=numppl)
    gmm.fit(datapts)
    #predictions from gmm
    labels = gmm.predict(datapts)
    a_set = set(labels)
    number_of_unique_values = len(a_set)
    #print("Number of people: ", number_of_unique_values)
    plt.scatter(datapts[:, 0], datapts[:, 1], s=8, c=labels, cmap='rainbow')

    # get "centroids" - samples with greatest probability
    centers = np.empty(shape=(gmm.n_components, datapts.shape[1]))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i], allow_singular=True).logpdf(datapts)
        centers[i, :] = datapts[np.argmax(density)]
    plt.scatter(centers[:, 0], centers[:, 1], s=20, color='black')
    plt.ylim([0, 3600])
    plt.xlim([0, 1200])
    plt.title('GMM Clustering')
    plt.show()
    #print(centers.tolist())
    return centers

def get_results(GTcenters, yolocenters):
    # Get results
    results = [] # numTP, numFP, numFN, moda, modp, precision, recall
    detectall_mapped = []  # all TP matched pairs for all frames
    groundtruthall_mapped = []
    for frame in range(len(GTcenters)):
        detect = np.array(yolocenters["frame{:d}".format(frame)])
        groundtruth = np.array(GTcenters["frame{:d}".format(frame)])
        total, TP, FP, FN, TP_det, TP_gt = hungarian_matching(groundtruth, detect, 50, verbose=False) #radius_match = 50
        numTP = len(TP)
        numFP = len(FP)
        numFN = len(FN)
        #print(total, "\n TP", numTP, "\n FP", numFP, "\n FN", numFN)
        numGT = len(groundtruth) # number of ground truth in frame

        moda = MODA(numFN, numFP, groundtruth)
        modp = MODP(TP_det, TP_gt)
        prec = precision(numTP, numFP)
        rec = recall(numTP, numFN)
        #print('\nMODA', moda, '\nMODP', modp, '\nPrecision', prec, '\nRecall', rec)
        results.append([numTP, numFP, numFN, moda, modp, prec, rec, numGT])
        detectall_mapped.append(TP_det)
        groundtruthall_mapped.append(TP_gt)

    results = np.array(results)
    fnall = results[:,2]
    fpall = results[:,1]
    gtall = results[:,7]
    print(len(gtall))
    nmoda = NMODA(fnall, fpall, gtall)
    nmodp = NMODP(detectall_mapped, groundtruthall_mapped)
    print("\nNMODA", nmoda, "\nNMODP", nmodp)
    return results, nmoda, nmodp


def MODA_plot(GTcenters, yolocenters):
    # get MODA against radius plot
    av_moda = []
    rlist = [30, 50, 70, 90, 110]  # vary radius for assigning detections to ground truth

    for r in rlist:
        rm = []
        for frame in range(len(GTcenters)):
            detect = np.array(yolocenters["frame{:d}".format(frame)])
            groundtruth = np.array(GTcenters["frame{:d}".format(frame)])
            t, tp, fp, fn, tpdet, tpgt = hungarian_matching(groundtruth, detect, r, verbose=False)  # radius_match = 50
            numtp = len(tp)
            numfp = len(fp)
            numfn = len(fn)
            rmoda = MODA(numfn, numfp, groundtruth)
            rm.append(rmoda)
        av_moda.append(np.mean(rm))

    plt.scatter(rlist, av_moda)
    plt.plot(rlist, av_moda)
    plt.xlabel('radius (cm)')
    plt.ylabel('MODA')
    plt.show()
    return

#-----------------------------------------------------------------------------------

with open('D:/Programming/CVProject/CVProject/branch/GTcenters', 'rb') as f:
    GTcenters = pickle.load(f)

with open('D:/Programming/CVProject/CVProject/branch/GTproj', 'rb') as f:
    GTproj = pickle.load(f)

with open('D:/Programming/CVProject/CVProject/branch/yolocenters', 'rb') as f:
    yolocenters = pickle.load(f)

with open('D:/Programming/CVProject/CVProject/branch/yoloproj', 'rb') as f:
    yoloproj = pickle.load(f)

with open('D:/Programming/CVProject/CVProject/branch/yolocenterssubset', 'rb') as f:
    yolosubset = pickle.load(f)

results, nmoda, nmodp = get_results(GTcenters, yolocenters)

# save results
# convert array into dataframe
DF = pd.DataFrame(results)
# save the dataframe as a csv file
DF.to_csv("evalsub_results.csv")

# visualization
MODA_plot(GTcenters, yolocenters)

# yolo projections with ground truth centers
plt.scatter(yoloproj["frame0"][:,5], yoloproj["frame0"][:,6], s = 3)
plt.scatter(GTcenters["frame0"][:,0], GTcenters["frame0"][:,1], s = 3, color='red')
plt.title('Detection Projections and Annotated Positions')
plt.show()

# GMM plot
ic = int((len(yoloproj["frame0"]) / 7)) # average number of people
#print(ic)
detectpts = np.array([yoloproj["frame0"][:, 5], yoloproj["frame0"][:, 6]])
centers = GMM(detectpts, ic)




