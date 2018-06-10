# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""This module is used to perform experiments on Anomaly Detection on Subjects.
Specifically, one subject (referrer) is used as a template, real, genuine and an other subject (tester)
is used as a tester. The tester can be either an impostor or a genuine.
For example, choose for subject1= 'user1' and for subject2= 'user2' and perform
the test. If subject2 passes, we have a False Accept. Now if subject2 = 'user1'
and do not pass, we have a False Reject"""
import extract
import general_purpose
import numpy as np
import time
from copy import deepcopy
from random import randint
import itertools
import datetime
# sklearn
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
# Plot Stuff
from skimage import measure
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib as mpl


def ret_digraph_points(sed, digraph):
    """Finds the digraph points of the subject extracted data.
    Parameters
    ----------
    `sed` (object) "_subject","_track_code", "data": [{"digraph","points"}]
    Returns
    ---------
    (list) The points of the particular digraph found
    """
    ret = [d['points'] for d in sed['data'] if d['digraph'] == digraph]
    if ret == []:
        # print '**Warning: No digraph points found in ret_digraph_points, digraph:' + digraph
        _foo = 1
    else:
        ret = ret[0]
    return ret


def remove_missing_values(events):
    """Removes missing values from events. For example if a keyUp is missing or keyDown etc"""
    ret = deepcopy(events)
    srchd, key_events = [], []
    for evt in events:
        _tmp = [(j, e) for j, e in enumerate(events) if e['key']
                == evt['key'] and not e['key'] in srchd]
        if _tmp != []:
            key_events.append(_tmp)
        srchd.append(evt['key'])
    dels = []
    for di_evts in key_events:
        if di_evts[0][1]['event'] == 'keystrokeUp':
            dels.append(di_evts[0][0])
        if di_evts[len(di_evts) - 1][1]['event'] == 'keystrokeDown':
            dels.append(di_evts[len(di_evts) - 1][0])
    if dels != []:
        for i in sorted(dels, reverse=True):
            del ret[i]
    return ret


def svm_add_2d_hyperplane(model, ax, plotted_points):
    """Adds a 2d hyperplane to the plot."""
    X_MIN = np.min(plotted_points[:, 0])
    X_MAX = np.max(plotted_points[:, 0])
    Y_MIN = np.min(plotted_points[:, 1])
    Y_MAX = np.max(plotted_points[:, 1])
    # plot the line, the points, and the nearest vectors to the plane
    xx, yy = np.mgrid[X_MIN:X_MAX:200j, Y_MIN:Y_MAX:200j]
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plot.contourf(xx, yy, Z, levels=np.linspace(
        Z.min(), 0, 7), cmap=plot.cm.PuBu)
    a = plot.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plot.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    return a.collections[0]


def svm_add_3d_hyperplane(model, ax, plotted_points):
    """Adds a 3d hyperplane to plot"""
    SPACE_SAMPLING_POINTS = 70
    X_MIN = np.min(plotted_points[:, 0])
    X_MAX = np.max(plotted_points[:, 0])
    Y_MIN = np.min(plotted_points[:, 1])
    Y_MAX = np.max(plotted_points[:, 1])
    Z_MIN = np.min(plotted_points[:, 2])
    Z_MAX = np.max(plotted_points[:, 2])
    xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                             np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                             np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))
    if hasattr(model, 'decision_function'):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    elif hasattr(model, 'predict_proba'):
        Z = model.predict_proba(
            np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
    else:
        exit('No decision function or predict_proba for classifer')
    Z = Z.reshape(xx.shape)
    verts, faces, _, _ = measure.marching_cubes(Z, 0)
    verts = verts * \
        [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
    verts = verts + [X_MIN, Y_MIN, Z_MIN]
    mesh = Poly3DCollection(verts[faces],
                            facecolor='orange', edgecolor='gray', alpha=0.4)
    ax.add_collection3d(mesh)


def svm_plot_results(model, train, _test, labels_test_pred, plt_title_info={}):
    """TODO"""
    ALPHA = 0.9
    # Init
    fig = plot.figure()
    _3d = True if train.shape[1] == 3 else False
    ax = fig.add_subplot(
        1, 1, 1, projection='3d') if _3d is True else fig.add_subplot(1, 1, 1)

    leg_ax, leg_str = (), ()
    # Generate hyperplane
    if _3d is True:
        svm_add_3d_hyperplane(model, ax, np.append(train, _test, axis=0))
    else:
        hyper = svm_add_2d_hyperplane(
            model, ax, np.append(train, _test, axis=0))
        leg_ax += (hyper,)
        leg_str += ('Learned Frontier',)

    # Plot train
    axtr = ax.scatter(train[:, 0], train[:, 1], train[:, 2], color='red',
                      s=80, edgecolor='k', alpha=ALPHA) if _3d is True else ax.scatter(train[:, 0], train[:, 1], color='red', s=80, edgecolor='k', alpha=ALPHA)
    leg_ax += (axtr,)
    leg_str += (plt_title_info['referrer'] + ' Train Data',)

    # Plot Test Considered as Outliers
    pred_outliers = np.array(
        [_test[i] for i, l in enumerate(labels_test_pred) if l == -1])
    if len(pred_outliers) > 0:
        axtstout = ax.scatter(pred_outliers[:, 0], pred_outliers[:, 1], pred_outliers[:, 2], color='black',
                              s=80, alpha=ALPHA) if _3d is True else ax.scatter(pred_outliers[:, 0], pred_outliers[:, 1], color='black', s=80, alpha=ALPHA)
        leg_ax += (axtstout,)
        leg_str += (plt_title_info['tester'] + ': Predicted as Outliers (%d/%d)' % (
            len(pred_outliers), len(_test)),)

    # Plot Test Considered as Inlieers
    pred_inliers = np.array([_test[i]
                             for i, l in enumerate(labels_test_pred) if l == 1])
    if len(pred_inliers) > 0:
        axtstin = ax.scatter(pred_inliers[:, 0], pred_inliers[:, 1], pred_inliers[:, 2], color='green',
                             s=80, edgecolor='k', alpha=ALPHA) if _3d is True else ax.scatter(pred_inliers[:, 0], pred_inliers[:, 1], color='green', s=80, edgecolor='k', alpha=ALPHA)
        leg_ax += (axtstin,)
        leg_str += (plt_title_info['tester'] +
                    ': Predicted as Inliers (%d/%d)' % (
            len(pred_inliers), len(_test)),)

    # Title and labels
    ax.set_title('Digraph: ' + plt_title_info['digraph'])
    ax.legend(leg_ax, leg_str, loc='upper right')  # upper right

    plot.show()


def gmm_make_ellipses(gmm, ax):
    """Plots the GMM components with ellipses"""
    COLORS = ['darkred', 'red', '#f66767']  # third is lightred
    for i in range(len(gmm.weights_)):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[i][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[i]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[i, :2], v[0], v[1],
                                  180 + angle, color=COLORS[i])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def gmm_plot_results(model, train, test, labels_test_pred=[], plt_title_info={}):
    """Plots gmm results"""
    ALPHA = 1
    COLOR_TRAIN = 'red'
    COLOR_TEST = 'blue'
    print plt_title_info
    # Init
    fig = plot.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Plot Train
    ax.scatter(train[:, 0], train[:, 1], color=COLOR_TRAIN, s=80,
               alpha=ALPHA, label=plt_title_info['referrer'] + ' Train Data')
    # Plot Gaussian Ellipses
    gmm_make_ellipses(model, ax)
    # Plot Test
    ax.scatter(test[:, 0], test[:, 1], color=COLOR_TEST, s=80,
               alpha=ALPHA, label=plt_title_info['tester'] + ' Test Data')
    ax.set_title('Digraph: ' + plt_title_info['digraph'])
    plot.legend(loc='upper right')
    plot.show()


def digraph_test(ref_points, test_points, algorithm, algo_params={}, visualize_results=False, plt_title_info={}):
    """Tests digraph test_points against digraph ref_points. \n
    Performs novelty detection with templates the ref_points and the test data the test_points
    Parameters
    ----------
    `ref_points` (numpy.array) Nx3 numpy array with points of referrer\n
    `test_points` (numpy.array) Nx3 numpy array with points of tester\n
    `algorithm` (string) The algorithm to be used (OneClassSVM or GMM)\n
    `algo_params` (dict) Some paramaetrs for the algo to be used
    `visualize_results` (boolean) IF you want to plot reslts
    `plt_title_info` (dict) Some info for the plot
    Returns
    ----------
    (int) The number of points from test_points that were considered as INLIERS from the algorithm
    """
    # If tester is actually the referrer, remove the test points from train ref_data
    # to make prediction more fair.
    # if plt_title_info['referrer'] == plt_title_info['tester']:
    #     _tmp = np.array([row for row in ref_points if row not in test_points])
    #     ref_points = _tmp
    # Fit ref
    model = globals()[algorithm](
        nu=0.15, gamma=algo_params['gamma']).fit(ref_points)

    # Predict test
    pred_labels = model.predict(test_points)

    # Show results if needed
    if visualize_results is True:
        svm_plot_results(model, ref_points, test_points,
                         pred_labels, plt_title_info)

    # Return count_global of predicted inliers
    return pred_labels[pred_labels == 1].size


def compute_malahanobis(mu, S, x):
    """Computes malahanobis distance of point x(1xD, D dimensions) from the distribution with means and covs"""
    return np.sqrt(np.matmul(np.matmul((x - mu), np.linalg.inv(S)), np.transpose(x - mu)))


def digraph_test_GMM(ref_points, test_points, algo_params={},  visualize_results=False, plt_title_info={}):
    """Perform anomaly detection, with Gaussian Mixture Components according to my algorithm.
    Returns
    -------
    (list 1xM where M number of components) The weighted number of predicted inliers for each component
    """
    N_COMPONENTS = algo_params['n_components']
    DELTA = algo_params['delta']

    # Construct the cluster(s)
    gmm_model = GaussianMixture(n_components=N_COMPONENTS)
    gmm_model.fit(ref_points)
    weights = gmm_model.weights_
    means = gmm_model.means_
    covs = gmm_model.covariances_

    # Get the number of points for each cluster of trained data.
    train_labels = gmm_model.predict(ref_points)
    # stds = []
    # # Reshape covs and get stds
    # for c in gmm_model.covariances_:
    #     vec = np.sqrt(np.array([[c[i, i] for i in range(c.shape[0])]]))
    #     vec = np.array([[c[i, i] for i in range(c.shape[0])]])
    #     if stds == []:
    #         stds = vec
    #     else:
    #         stds = np.append(stds, vec, axis=0)

    # Compute score for each component
    score_comp = N_COMPONENTS * [0.]
    for m in range(N_COMPONENTS):

        # If a cluster has a limited number of points, ignore it
        if len([l for l in train_labels if l == m]) < 2:
            continue

        # Test each point
        _pass = 0.
        for test_point in test_points:

            # With malahnobis
            mlh = mahalanobis(test_point, means[m], np.linalg.inv(covs[m]))
            if mlh <= DELTA:
                _pass += 1

            # My first way:
            # flag = True
            # for dim, val in enumerate(test_point):
            #     if not (val >= means[m, dim] - DELTA * stds[m, dim] and val <= means[m, dim] + DELTA * stds[m, dim]):
            #         flag = False
            #         break
            # if flag is True:
            #     _pass += 1

        # Scale with the weight of component
        score_comp[m] += weights[m] * _pass

    # Show results if needed
    if visualize_results is True:
        if ref_points.shape[1] != 2:
            exit('Warning! Data is not 2d to plot the GMM ellipses!')
        gmm_plot_results(gmm_model, ref_points, test_points,
                         plt_title_info=plt_title_info)

    # Return score
    return score_comp


def test(referrer, tester, test_word, algorithm, algo_params={}, visualize_results=False, apply_PCA=False,  clear_referrer_with_algo=False, clean_algo_parameters={"name": 'EllipticEnvelope', "contamination": 0.1, "visualize_results": False}):
    """Tests tester against referrer.

    The tester is the subject which claims to be the referrer. So
    he can either tell truth(genuine) or lie(impostor). This function
    will determine if the tester typing pattern looks a lot like
    the referrers' typing pattern.
    Parameters
    ----------
    `referrer` (object) {"_subject", "_track_code", "data": [{"digraph", "points"}]}
        It contains the extracted data timings for each digraph for the referrer\n
    `tester` (list) Same style as referrer
    `test_word` (string) The test word
    `visualize_results` (boolean) Plot the results
    `apply_PCA` (boolean) Apply pca before test
    `clean_referrer_with_algo` (boolean) If you want to clean referrer data with outlier detection methods before testing
    `algo_clean_parameters` (object) "name", "contamination", "visualize_results"
    Returns
    ---------
    The percentage of digraph points that were considered INLIERS.
    """

    # Split the test_word to digraphs
    test_digraphs = [('Key' + c[0].upper() if c[0] != ' ' else 'Space') + ('Key' + c[1].upper() if c[1] != ' ' else 'Space')
                     for c in [z[0] + z[1] for z in zip(test_word, test_word[1:])]]

    # Test each digraph data of tester against referrer
    count_insufficient_ref_samples = 0.
    # `score` Represents number of inliers if algo != 'GMM' else represents score for each GMM component
    score = 0. if algorithm != 'GMM' else []
    count_global = 0.
    for test_digraph in test_digraphs:

        # Find the digraph data of referrer
        ref_di_points = ret_digraph_points(
            referrer, test_digraph).astype(float)
        # print ref_di_points
        if len(ref_di_points) >= 10:

            # If tester and referrer sample
            if referrer['_subject'] == tester['_subject'] and referrer['_track_code'] == tester['_track_code']:
                _tmp = ref_di_points
                np.random.shuffle(_tmp)
                perc = int(np.floor(0.8 * len(_tmp)))
                ref_di_points = _tmp[0:perc, :]
                test_di_points = _tmp[perc:, :]
            else:
                # Sample some data for the particular digraph in tester
                _test_di_points = ret_digraph_points(tester, test_digraph)
                test_di_points = _test_di_points[np.random.choice(
                    _test_di_points.shape[0], size=randint(10, 12), replace=False), :]

            # Transform ref and tester data with standard scaler
            sscaler = StandardScaler(with_mean=True, with_std=True).fit(
                np.append(ref_di_points, test_di_points, axis=0))
            train_points = sscaler.transform(ref_di_points)
            test_points = sscaler.transform(test_di_points)

            # Apply PCA if needed
            if apply_PCA is True:
                pca_model = PCA().fit(
                    np.append(train_points, test_points, axis=0))
                train_points = pca_model.transform(train_points)[:, 0:2]
                test_points = pca_model.transform(test_points)[:, 0:2]

            # Clean referrer points from noise if needed
            if clear_referrer_with_algo is True:
                train_points = general_purpose.clean_with_algo(
                    train_points, algorithm=clean_algo_parameters['name'], contamination=clean_algo_parameters['contamination'])

            # Anomaly-Test the digraph and get the count_global of inliers
            if algorithm != 'GMM':
                score += digraph_test(train_points,
                                      test_points, algorithm, algo_params, visualize_results=visualize_results,
                                      plt_title_info={"referrer": referrer['_subject'],
                                                      "tester": tester['_subject'], "digraph": test_digraph})
            else:
                # Special procedure for GMM
                _sc = digraph_test_GMM(train_points,
                                       test_points, algo_params, visualize_results=visualize_results,
                                       plt_title_info={"referrer": referrer['_subject'],
                                                       "tester": tester['_subject'], "digraph": test_digraph})
                score.append(_sc)

        else:

            count_insufficient_ref_samples += len(test_points)

        count_global += len(test_points)

    # Find eventually the total score
    if algorithm != 'GMM':
        total_score = score / \
            (count_global - count_insufficient_ref_samples)
    else:
        total_score = 0.
        for s in score:
            total_score += sum(s)
        total_score = total_score / \
            (count_global - count_insufficient_ref_samples)

    # print '\n'
    # print 'Digraph passed: ' + str(total_score) + '%, ' + str(int(score)) + ' out of ' + str(int(count_global - count_insufficient_ref_samples))
    # print 'Insufissient digraph samples: ' + str(count_insufficient_ref_samples) + ' out of ' + str(int(count_global))
    # print total_score
    if count_insufficient_ref_samples > 0:
        print 'Insuffisient samples: %d' % (count_insufficient_ref_samples)
        print 'For word: ' + test_word
    return total_score


def experiment(sed_list, test_words, trials=100, testing_threshold=0.5, apply_PCA=False, visualize_results=False, algorithm='OneClassSVM', algo_params={}, print_results=True, dataset='MIXED', clean_referrer_with_algo=True, write_to_txt=True, results_filename_txt='anomaly-results'):
    """Experiment for anomaly detection test.\n
        Specifically, it picks randomly (or not), two subjects from the sed_list. One will be the trained_template
        and the other will be the tester_template. Then for each digraph in digraphs pool it performs test.

    Parameters
    ----------
    `sed_list` (list) A list of dictionaries with "_subject", "_track_code" and 'data':[{"points", "digraph"}].
        If random is False then the first two subjects of sed_list are picked, with the referrer_index to define the referrer\n
    `test_words` (list) A list with test_words
    `trials` (int) How many trials \n
    `random` (boolean) If False then you must specify two subjects in the sed_list \n
    `referrer_index` (int) Default 0 \n
    `apply_pca` (boolean) Default False. IF you want to apply PCA prior to modelling\n
    Returns
    ---------
    `FAR` (float) False accept rate \n
    `FRR` (float) False Reject Rate
    """
    if print_results is True:
        print '\nPerforming Anomaly Detection Experiment...'
    start = time.time()
    if not isinstance(test_words, list):
        test_words = [test_words]
    # Catch wrong parametrs
    if len(sed_list) < 2:
        exit('**In function anomaly.experiment: Length of sed_list <2')

    # Make all possible pairs from subjects (including pair with same item)
    subjects_all_pairs = list(
        itertools.combinations_with_replacement(sed_list, 2))

    count_false_accepts = 0.
    count_false_rejects = 0.
    count_t = 0.  # counts the tests happened
    # Repeat this many trials
    for trial in range(trials):
        # For each word
        for word in test_words:

            # For each possible pair
            for subject_pair in subjects_all_pairs:

                # For each senario
                _r = 1 if subject_pair[0]['_subject'] == subject_pair[1][
                    '_subject'] and subject_pair[0]['_track_code'] == subject_pair[1]['_track_code'] else 2

                for ref_index in range(_r):

                    # Pick referrer and tester
                    referrer = subject_pair[ref_index]
                    tester = subject_pair[1] if ref_index == 0 else subject_pair[0]
                    is_tester_impostor = False if referrer['_subject'] == tester[
                        '_subject'] and referrer['_track_code'] == tester['_track_code'] else True

                    # Perform test to word
                    total_score = test(referrer, tester, word, algorithm, algo_params,
                                       apply_PCA=apply_PCA, clear_referrer_with_algo=clean_referrer_with_algo, visualize_results=visualize_results)
                    count_t += 1
                    # Check if it passes threshold
                    if total_score >= testing_threshold:
                        # Tester was considered as genuine
                        if is_tester_impostor is True:
                            count_false_accepts += 1
                    else:
                        # Tester was considered as impostor
                        if is_tester_impostor is False:
                            count_false_rejects += 1

    # Calculate total FAR and FRR
    FAR = count_false_accepts / count_t
    FRR = count_false_rejects / count_t

    # Save results
    resstr = '\n-------------------------------------------'
    now = str(datetime.datetime.now())[:-7]
    resstr += '\n# Date: ' + now
    resstr += '\n# Wordset: ' + str(test_words)
    resstr += '\n# Dataset: ' + str(dataset)
    resstr += '\n# Subjects: ' + str([s['_subject'] for s in sed_list])
    resstr += '\n# Trials: ' + str(trials)
    resstr += '\n# Data whiten prior: ' + str(clean_referrer_with_algo)
    resstr += '\n# PCA: ' + str(apply_PCA)
    resstr += '\n# Testing Threshold: ' + str(testing_threshold)
    resstr += '\n# Algorithm: ' + algorithm
    resstr += '\n# Algorithm Parameters: ' + str(algo_params)
    resstr += '\nResults:'
    resstr += '\n--> FAR: %.2f%%' % (100 * FAR)
    resstr += '\n--> FRR: %.2f%%' % (100 * FRR)
    resstr += '\nFinished in %.2fs' % (time.time() - start)
    resstr += '\n-------------------------------------------\n\n'
    if write_to_txt is True:
        with open(results_filename_txt + '.txt', 'a') as fin:
            fin.write(resstr)
    if print_results is True:
        print resstr

    return (FAR, FRR)
