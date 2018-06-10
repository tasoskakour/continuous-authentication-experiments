# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""This module is used to experiment on classification on subjects data.
    Generally, given two subjects, it will try to separate them, and predict
    correctly the test set of each subject
"""
import numpy as np
import general_purpose
import datetime
import time
import itertools
# Machine Learning Stuff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import recall_score, f1_score, roc_curve, roc_auc_score
# Plot stuff
from skimage import measure
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.font_manager
import matplotlib.patches as mpatches


def add_2d_hyperplane(model,  plotted_points):
    """Adds a 2d hyperplane to the plot."""
    X_MIN = np.min(plotted_points[:, 0])
    X_MAX = np.max(plotted_points[:, 0])
    Y_MIN = np.min(plotted_points[:, 1])
    Y_MAX = np.max(plotted_points[:, 1])
    XX, YY = np.mgrid[X_MIN:X_MAX:200j, Y_MIN:Y_MAX:200j]
    if hasattr(model, 'decision_function'):
        Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()])
    elif hasattr(model, 'predict_proba'):
        Z = model.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:, 1]
    else:
        exit('No decision function or predict_proba for classifer')
    # print Z
    Z = Z.reshape(XX.shape)
    plot.pcolormesh(XX, YY, Z > 0, cmap='RdBu', alpha=0.4)
    if model.__class__.__name__ == 'SVC':
        plot.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                     linestyles=['--', '-', '--'], levels=[-.5, 0, .5])


def add_3d_hyperplane(model, ax, plotted_points):
    """Adds a 3d hyperplane, based on desicion function of model"""
    SPACE_SAMPLING_POINTS = 70

    # Define the size of the space which is interesting
    X_MIN = np.min(plotted_points[:, 0])
    X_MAX = np.max(plotted_points[:, 0])
    Y_MIN = np.min(plotted_points[:, 1])
    Y_MAX = np.max(plotted_points[:, 1])
    Z_MIN = np.min(plotted_points[:, 2])
    Z_MAX = np.max(plotted_points[:, 2])

    # Generate a regular grid to sample the 3D space for various operations later
    xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                             np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                             np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))

    # Calculate the distance from the separating hyperplane of the SVM for the
    # whole space using the grid defined in the beginning
    if hasattr(model, 'decision_function'):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    elif hasattr(model, 'predict_proba'):
        Z = model.predict_proba(
            np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
    else:
        exit('No decision function or predict_proba for classifer')
    Z = Z.reshape(xx.shape)

    # Plot the separating hyperplane by recreating the isosurface for the distance
    # == 0 level in the distance grid computed through the decision function of the
    # SVM. This is done using the marching cubes algorithm implementation from
    # scikit-image.
    verts, faces, _, _ = measure.marching_cubes(Z, 0)
    # Scale and transform to actual size of the interesting volume
    verts = verts * \
        [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
    verts = verts + [X_MIN, Y_MIN, Z_MIN]
    # and create a mesh to display
    mesh = Poly3DCollection(verts[faces],
                            facecolor='orange', edgecolor='gray', alpha=0.4)
    ax.add_collection3d(mesh)


def add_scatter(ax, points, labels=[], labels_paint={"colors": [], "markers": []}, color='', marker='o'):
    """Scatter plot the points.
    Parameters
    ----------
    `ax` The subplot model\n
    `points` (np.array) of Nx3 or Nx2\n
    `labels` Optional: list with true labels of points\n
    `labels_paint` Optional: Parameters\n
    `color` (str) Color if labels==[]\n
    `marker` (str) The marker\n
    Returns
    -------
    The scatter plot if labels ==[] else The scatter plots along with lengths
    """
    S = 90
    EDGECOLOR1 = 'k'
    EDGECOLOR2 = 'w'
    EDGECOLOR3 = 'darkgreen'
    _3d = True if points.shape[1] == 3 else False
    if labels == []:
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=S, color=color, alpha=0.9, edgecolors=EDGECOLOR1) if _3d is True else ax.scatter(
            points[:, 0], points[:, 1], s=S, color=color, marker=marker, alpha=0.9, edgecolors=EDGECOLOR1)
        return p
    else:
        # Separate points by labels
        points0 = np.array(
            [points[i] for i, l in enumerate(labels) if l == 0])
        points1 = np.array(
            [points[i] for i, l in enumerate(labels) if l == 1])
        lenp0, lenp1 = len(points0), len(points1)
        if lenp0 == 0:
            points0 = np.array([[np.nan, np.nan, np.nan]]) if _3d is True else np.array(
                [[None, None]])
        if lenp1 == 0:
            points1 = np.array([[np.nan, np.nan, np.nan]]) if _3d is True else np.array(
                [[None, None]])

        # Add scatter with diff color and marker
        p0 = ax.scatter(points0[:, 0], points0[:, 1], points0[:, 2], s=S, color=labels_paint['colors'][0], marker=labels_paint['markers'][0], alpha=0.9, edgecolor=EDGECOLOR3, linewidth=3) if _3d is True else ax.scatter(
            points0[:, 0], points0[:, 1], s=S, color=labels_paint['colors'][0], marker=labels_paint['markers'][0], alpha=0.9, edgecolors=EDGECOLOR2, linewidth='3')
        p1 = ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=S, color=labels_paint['colors'][1], marker=labels_paint['markers'][1], alpha=0.9, edgecolor=EDGECOLOR3, linewidth=3) if _3d is True else ax.scatter(
            points1[:, 0], points1[:, 1], s=S, color=labels_paint['colors'][1], marker=labels_paint['markers'][1], alpha=0.9, edgecolors=EDGECOLOR2, linewidth='3')
        return p0, lenp0, p1, lenp1


def apply_algos(algorithms, X, Y, visualize_results=False, fig_suptitple=''):
    """Apply classification algorithm to X and Y
    X is positive Y is negative class
    Parameters
    ----------
    \n `X` (np.array) The data of class 1
    \n `Y` (np.array) The data of class 2
    \n `visualize_results` (boolean) If you want to plot results
    \n `plot_ROC` (boolean) If you want to plot ROC
    Returns
    ---------
    (dict) `{"SCORE", "F_Measure_+", "F_Measure_-", "TPR", "TNR", "EER", "AUC"}`
    """
    split_perc = 0.8

    # Split Class 1 to train and test
    np.random.shuffle(X)
    X_train = X[0:int(np.ceil(split_perc * len(X))), :]
    X_test = X[int(np.ceil(split_perc * len(X))):, :]

    # Split Class 2 to train and test
    np.random.shuffle(Y)
    Y_train = Y[0:int(np.ceil(split_perc * len(Y))), :]
    Y_test = Y[int(np.ceil(split_perc * len(Y))):, :]

    # Merge data, construct labels
    merged_train = np.append(X_train, Y_train, axis=0)
    labels_train_true = len(X_train) * [0] + len(Y_train) * [1]
    merged_test = np.append(X_test, Y_test, axis=0)
    labels_test_true = len(X_test) * [0] + len(Y_test) * [1]

    # Dict all results
    labels_test_pred = {}
    model = {}
    fpr_pts, tpr_pts = {}, {}
    F_Measure_Positive, F_Measure_Negative, TPR, TNR, SCORE, EER, AUC = {}, {}, {}, {}, {}, {}, {}

    # Apply algorithms
    for algorithm in algorithms:

        # Apply and fit model
        if hasattr(globals()[algorithm](), 'kernel') and hasattr(globals()[algorithm](), 'gamma'):
            model[algorithm] = globals()[algorithm](kernel='rbf', gamma='auto')
        elif hasattr(globals()[algorithm](), 'n_neighbors'):
            model[algorithm] = globals()[algorithm](n_neighbors=5)
        else:
            model[algorithm] = globals()[algorithm]()
        model[algorithm].fit(merged_train, labels_train_true)

        # Predict
        labels_test_pred[algorithm] = model[algorithm].predict(merged_test)

        # Calculate Metrics
        F_Measure_Positive[algorithm] = f1_score(
            labels_test_true, labels_test_pred[algorithm], pos_label=1)
        F_Measure_Negative[algorithm] = f1_score(
            labels_test_true, labels_test_pred[algorithm], pos_label=0)
        TPR[algorithm] = recall_score(
            labels_test_true, labels_test_pred[algorithm], pos_label=1)
        TNR[algorithm] = recall_score(
            labels_test_true, labels_test_pred[algorithm], pos_label=0)
        SCORE[algorithm] = model[algorithm].score(
            merged_test, labels_test_true)
        if hasattr(model[algorithm], 'decision_function'):
            scores = model[algorithm].decision_function(merged_test)
        elif hasattr(model[algorithm], 'predict_proba'):
            scores = model[algorithm].predict_proba(merged_test)[:, 1]
        else:
            exit('No decision_function or predict_proba for model')
        fpr_pts[algorithm], tpr_pts[algorithm], _ = roc_curve(
            labels_test_true, scores, pos_label=1)
        fnr_pts = 1 - tpr_pts[algorithm]
        EER[algorithm] = (fpr_pts[algorithm][np.nanargmin(np.absolute((fnr_pts - fpr_pts[algorithm])))] +
                          fnr_pts[np.nanargmin(np.absolute((fnr_pts - fpr_pts[algorithm])))]) / 2
        AUC[algorithm] = roc_auc_score(labels_test_true, scores)

    # Show results if needed
    # One figure for classification results, one algo in each subplot
    # One figure for ROC curves, one roc algo curve in each subplot
    if visualize_results is True:

        if len(algorithms) == 1:
            pltdim1, pltdim2 = 1, 1
        elif len(algorithms) == 2:
            pltdim1, pltdim2 = 1, 2
        else:
            pltdim1, pltdim2 = np.ceil(np.sqrt(len(algorithms))), np.ceil(
                np.sqrt(len(algorithms)))
        _3d = True if X.shape[1] == 3 else False

        # Init Figure for class results
        fig = plot.figure()
        fig.suptitle(fig_suptitple)

        for i, algorithm in enumerate(algorithms):

            # Add subplot
            ax = fig.add_subplot(
                pltdim1, pltdim2, i + 1, projection='3d') if _3d is True else fig.add_subplot(pltdim1, pltdim2, i + 1)

            # Add the separating hyperplane
            if _3d is True:
                if algorithm == 'SVC':
                    add_3d_hyperplane(model[algorithm], ax,  np.append(
                        np.append(X_train, Y_train, axis=0), merged_test, axis=0))
            else:
                add_2d_hyperplane(model[algorithm], np.append(
                    np.append(X_train, Y_train, axis=0), merged_test, axis=0))

            # Scatter X Train and Y Train
            ax_X_train = add_scatter(ax, X_train, color='red', marker='o')
            ax_Y_train = add_scatter(ax, Y_train, color='blue', marker='o')
            leg_tuple_ax = (ax_X_train, ax_Y_train)
            leg_tuple_str = ('Negative Class Training Data (%d)' % (len(X_train)),
                             'Positive Class Training Data (%d)' % (len(Y_train)))

            # Scatter TN (Xclass) and FP
            labels_X_test_pred = labels_test_pred[algorithm][0:len(X_test)]
            ax_TN, lenTN, ax_FP, lenFP = add_scatter(ax, X_test, labels=labels_X_test_pred, labels_paint={
                "colors": ['red', 'red'], "markers": ['o', 'o']})
            leg_tuple_ax += (ax_TN, ax_FP)
            leg_tuple_str += ('True Negatives (%d/%d)' % (lenTN, lenTN + lenFP),
                              'False Positives (%d/%d)' % (lenFP, lenTN + lenFP))

            # Scatter TP (Yclass) and FN
            labels_Y_test_pred = labels_test_pred[algorithm][len(X_test):]
            ax_FN, lenFN, ax_TP, lenTP = add_scatter(ax, Y_test, labels=labels_Y_test_pred, labels_paint={
                "colors": ['blue', 'blue'], "markers": ['o', 'o']})
            leg_tuple_ax += (ax_TP, ax_FN)
            leg_tuple_str += ('True Positives (%d/%d)' % (lenTP, lenTP + lenFN),
                              'False Negatives (%d/%d)' % (lenFN, lenTP + lenFN))

            # Set Title and Legend
            if algorithm == 'SVC':
                algorithm_name = 'SVM'
            elif algorithm == 'KNeighborsClassifier':
                algorithm_name = 'kNN'
            ax.set_title('Classifier: ' + algorithm_name)
            # ax.set_title('Classifier: ' + algorithm_name + '\nF-Measure_+ = %.2f, F-Measure_- = %.2f, TPR = %.2f, TNR = %.2f, EER = %.2f' %
            #              (F_Measure_Positive[algorithm], F_Measure_Negative[algorithm], TPR[algorithm], TNR[algorithm], EER[algorithm]))
            ax.legend(leg_tuple_ax, leg_tuple_str, loc=0,
                      bbox_to_anchor=(0, 0, 1, 0.93))
            # ax.legend(leg_tuple_ax, leg_tuple_str, loc='upper right')

        # Init Figure for ROCs
        fig = plot.figure()
        fig.suptitle(fig_suptitple)
        # Plot ROC Curve
        for i, algorithm in enumerate(algorithms):
            line = fig.add_subplot(pltdim1, pltdim2, i + 1)
            line.plot(fpr_pts[algorithm], tpr_pts[algorithm])
            line.plot([0, 1], [0, 1], 'k--')
            if algorithm == 'SVC':
                algorithm_name = 'SVM'
            elif algorithm == 'KNeighborsClassifier':
                algorithm_name = 'kNN'
            line.set_title(algorithm_name + '\nROC Curve, AUC = %.2f' %
                           (AUC[algorithm]))
            line.set_xlabel('FPR')
            line.set_ylabel('TPR')

        # Show dem figs
        plot.show()

    return {"SCORE": SCORE,
            "F_Measure_+": F_Measure_Positive,
            "F_Measure_-": F_Measure_Negative,
            "TPR": TPR,
            "TNR": TNR,
            "EER": EER,
            "AUC": AUC
            }


def experiment(subjects_extracted_list, algorithms, digraphs, n_trials, standard_scaler=True, apply_PCA=False, visualize_results=False,  algo_clean=False, algo_clean_parameters={"name": 'EllipticEnvelope', "contamination": 0.1, "visualize_results": False}, write_results_to_txt=True, dataset='MIXED'):
    """Applies classification experiment \n
    It picks randomly n digraphs (n=n_trials) from the pool of digraphs given, and for each digraph
    it performs classification to 2 randomly picked subjects from subjects_extracted_list
    Parameters
    ----------
    `subjets_extracted_list` (list) The list of dicts with subjects data\n
    `algorithms` (list) A list with algorithms as strings\n
    `digraphs` (list) The list of digraphs for this experiment\n
    `n_trials` (int) The number of trials for this experiment\n
    `standard_scaler` (boolean) If you want to apply scale before\n
    `apply_PCA` (boolean) If you want to apply PCA before\n
    `visualize_results` (boolean) If you want to visualize results of classification\n
    `algo_clean` (boolean) If you want to clean noise from data prior to classification\n
    `algo_clean_parameters` (dict) Some parameters for algo_clean\n
    `write_results_to_txt` (boolean) If you want to write results to txt\n
    `dataset` (str) Specify which dataset is used\n
    Returns
    ----------
    None
    """
    print 'Classification Experiment started.'
    start = time.time()
    if n_trials > 10 and visualize_results is True:
        exit('Wow... Visualize results with 10+ plots?')
    # Fix parameters
    if not isinstance(subjects_extracted_list, list):
        subjects_extracted_list = [subjects_extracted_list]
    if not isinstance(digraphs, list):
        digraphs = [digraphs]

    # Results
    avg_samps_len = 0.
    avg_pca_info_loss = 0.

    results = {"SCORE": dict((a, 0.) for a in algorithms),
               "F_Measure_+": dict((a, 0.) for a in algorithms),
               "F_Measure_-": dict((a, 0.) for a in algorithms),
               "TPR": dict((a, 0.) for a in algorithms),
               "TNR": dict((a, 0.) for a in algorithms),
               "EER": dict((a, 0.) for a in algorithms),
               "AUC": dict((a, 0.) for a in algorithms)}

    # Make all possible pairs from subjects
    subjects_all_pairs = list(
        itertools.combinations(subjects_extracted_list, 2))

    count_c = 0.  # counts how many classifications happened
    # Repeat this for n trials
    for trial in range(n_trials):

        # For each digraph
        for digraph in digraphs:

            # For each pair
            for subject_pair in subjects_all_pairs:

                # [ {"subject": '..', "points": [..]} , ...]
                subjects_table = []
                for se in subject_pair:

                    # Construct subjects_table with points and filter if needed
                    tmp = general_purpose.my_reshape(se, digraph)
                    if tmp != -1:
                        subjects_table.append(
                            general_purpose.my_reshape(se, digraph))
                    else:
                        print '***Warning: No samples found for digraph "' + digraph + '" of subject "' + se['_subject'] + '"'
                        continue

                avg_samps_len += sum([len(se['points'])
                                      for se in subjects_table]) / 2
                # Scale if needed
                if standard_scaler is True:
                    my_scaler = StandardScaler(with_mean=True, with_std=True).fit(
                        np.array([row for s in subjects_table for row in s['points']]))
                    for s in subjects_table:
                        s['points'] = my_scaler.transform(s['points'])
                # Apply PCA if needed
                if apply_PCA is True:
                    my_pca = PCA().fit(
                        np.array([row for s in subjects_table for row in s['points']]))
                    for s in subjects_table:
                        s['points'] = my_pca.transform(
                            s['points'])[:, 0:2]  # keep first 2 dims
                    pca_info_loss = my_pca.explained_variance_ratio_[2]
                    avg_pca_info_loss += pca_info_loss
                # Clean with algo if needed
                if algo_clean is True:
                    for s in subjects_table:
                        s['points'] = general_purpose.clean_with_algo(
                            s['points'], algorithm=algo_clean_parameters['name'], contamination=algo_clean_parameters['contamination'], visualize_results=algo_clean_parameters['visualize_results'])

                # If visualize_results is True fix suptitple
                suptitle = ''
                if visualize_results is True:
                    suptitle = ('Digraph: ' + digraph + '\nSubjects: ' +
                                subject_pair[0]['_subject'] + ', ' + subject_pair[1]['_subject'] + '\n')
                    if standard_scaler is True:
                        suptitle += 'Scaled'
                    if apply_PCA is True:
                        suptitle += ', PCA, Loss: %.0f%%' % (
                            100 * pca_info_loss)
                    # if algo_clean is True:
                    #     suptitle += ', Cleaned'

                # Apply The Classification algorithm
                res = apply_algos(algorithms, subjects_table[0]['points'], subjects_table[1]['points'],
                                  visualize_results=visualize_results, fig_suptitple=suptitle)
                count_c += 1

                # Update results
                for algorithm in algorithms:
                    results['SCORE'][algorithm] += res['SCORE'][algorithm]
                    results['F_Measure_+'][algorithm] += res['F_Measure_+'][algorithm]
                    results['F_Measure_-'][algorithm] += res['F_Measure_-'][algorithm]
                    results['TPR'][algorithm] += res['TPR'][algorithm]
                    results['TNR'][algorithm] += res['TNR'][algorithm]
                    results['EER'][algorithm] += res['EER'][algorithm]
                    results['AUC'][algorithm] += res['AUC'][algorithm]

    print 'Experiment Finished in %.2fs.' % (time.time() - start)

    # Save results
    for algorithm in algorithms:
        results['SCORE'][algorithm] = round(
            results['SCORE'][algorithm] / count_c, 2)
        results['F_Measure_+'][algorithm] = round(
            results['F_Measure_+'][algorithm] / count_c, 2)
        results['F_Measure_-'][algorithm] = round(
            results['F_Measure_-'][algorithm] / count_c, 2)
        results['TPR'][algorithm] = round(
            results['TPR'][algorithm] / count_c, 2)
        results['TNR'][algorithm] = round(
            results['TNR'][algorithm] / count_c, 2)
        results['EER'][algorithm] = round(
            results['EER'][algorithm] / count_c, 2)
        results['AUC'][algorithm] = round(
            results['AUC'][algorithm] / count_c, 2)

    now = str(datetime.datetime.now())[:-7]
    res_str = '# Date: ' + now
    res_str += '\n# Subjects: ' + \
        str(len(subjects_extracted_list)) + ' (' + dataset + ')'
    res_str += '\n# Type: All' if digraphs[0] == '' else '\n# Type: Digraphs = ' + str(
        digraphs)
    res_str += '\n# Avg Subject Samples: %.1f' % (
        avg_samps_len / count_c)
    res_str += '\n# Trials: ' + str(n_trials)
    res_str += '\n# Algorithm(s): ' + str(algorithms)
    res_str += '\n# Data Scaled prior: ' + str(standard_scaler)
    res_str += '\n# Data PCA prior: ' + str(apply_PCA)
    if apply_PCA is True:
        res_str += ', Loss: %.0f%%' % (100 *
                                       (avg_pca_info_loss / count_c))
    res_str += '\n# Data Whiten prior: ' + str(algo_clean)
    res_str += '\nScore: %s' % (results['SCORE'])
    res_str += '\nTPR: %s' % (results['TPR'])
    res_str += '\nF Measure+: %s' % (results['F_Measure_+'])
    res_str += '\nTNR: %s' % (results['TNR'])
    res_str += '\nF Measure-: %s' % (results['F_Measure_-'])
    res_str += '\nEER: %s' % (results['EER'])
    res_str += '\nAUC: %s' % (results['AUC'])
    res_str += '\n----------------------\n'
    # Print results either to txt or console
    if write_results_to_txt is True:
        with open('classification-results' + '.txt', 'a') as fin:
            fin.write(res_str)
    else:
        print res_str
