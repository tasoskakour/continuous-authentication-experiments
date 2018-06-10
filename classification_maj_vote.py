# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""
Classification using OneVsOne Reduction and Majority Vote
"""
import numpy as np
import general_purpose as gp
import datetime
import time
import itertools
from random import randint
# Machine Learning Stuff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def classifier_train(samples, labels, classifier, **kwargs):
    """Train the samples given their true labels with the given classifier. Returns the classifier model"""
    kwargs = kwargs['kwargs']
    if hasattr(globals()[classifier](), 'kernel') and hasattr(globals()[classifier](), 'gamma'):
        gamma = 'auto' if 'svc_gamma' not in kwargs else kwargs['svc_gamma']
        classifier_model = globals()[classifier](kernel='rbf', gamma=gamma)
    elif hasattr(globals()[classifier](), 'n_neighbors'):
        n_neighbors = 5 if 'knn_neighbors' not in kwargs else kwargs['knn_neighbors']
        classifier_model = globals()[classifier](n_neighbors=n_neighbors)
    else:
        classifier_model = globals()[classifier]()
    # print samples
    # print '\n'
    # print labels
    classifier_model.fit(samples, labels)
    return classifier_model


def classifier_predict(samples, classifier_model):
    """Predicts the samples given the classifier_model. Returns the predicted labels as array"""
    return classifier_model.predict(samples)


def word_test(sed_list, referrer_index, word, classifier_algorithm, apply_PCA, apply_clean=False, **kwargs):
    """Predicts which label from sed_list the word belongs. It uses OneVsOne Reduction Classification"""
    split_perc = 0.8

    # Make n(n-1)/2 pairs for OneVsOne Reduction
    subjects_indeces_all_pairs = list(
        itertools.combinations(range(len(sed_list)), 2))

    # Split the word/phrase to digraphs
    digraphs = [('Key' + c[0].upper() if c[0] != ' ' else 'Space') + ('Key' + c[1].upper() if c[1] != ' ' else 'Space')
                for c in [z[0] + z[1] for z in zip(word, word[1:])]]

    # The votes for each classifier for each digraph
    subjects_votes = len(sed_list) * [0]

    # Test each digraph
    for digraph in digraphs:

        # For the referrer split to train and test points
        # The test points will be tested from all classifiers later
        _ref_di_points = gp.ret_digraph_points(
            sed_list[referrer_index], digraph)
        np.random.shuffle(_ref_di_points)
        ref_di_points_train = _ref_di_points[0:int(
            np.floor(split_perc * len(_ref_di_points))), :]
        _ref_di_points_test = _ref_di_points[int(
            np.floor(split_perc * len(_ref_di_points))):, :]

        # For each subject pair
        for pair in subjects_indeces_all_pairs:

            # Get the di points for each subject for the particular digraph
            _s1_di_points, _s2_di_points = None, None
            if pair[0] == referrer_index:
                _s1_di_points = ref_di_points_train
            elif pair[1] == referrer_index:
                _s2_di_points = ref_di_points_train
            if _s1_di_points is None:
                _s1_di_points = gp.ret_digraph_points(
                    sed_list[pair[0]], digraph)
                np.random.shuffle(_s1_di_points)
                _s1_di_points = _s1_di_points[0:int(
                    np.floor(split_perc * len(_s1_di_points))), :]
            if _s2_di_points is None:
                _s2_di_points = gp.ret_digraph_points(
                    sed_list[pair[1]], digraph)
                np.random.shuffle(_s2_di_points)
                _s2_di_points = _s2_di_points[0:int(
                    np.floor(split_perc * len(_s2_di_points))), :]

            # Apply Scale
            sscaler = StandardScaler(with_mean=True, with_std=True).fit(
                np.append(np.append(_s1_di_points, _s2_di_points, axis=0), _ref_di_points_test, axis=0))
            s1_di_points = sscaler.transform(_s1_di_points)
            s2_di_points = sscaler.transform(_s2_di_points)
            ref_di_points_test = sscaler.transform(_ref_di_points_test)

            # Apply PCA if needed
            if apply_PCA is True:
                pca_model = PCA(n_components=2).fit(
                    np.append(np.append(s1_di_points, s2_di_points, axis=0), ref_di_points_test, axis=0))
                s1_di_points = pca_model.transform(s1_di_points)
                s2_di_points = pca_model.transform(s2_di_points)
                ref_di_points_test = pca_model.transform(ref_di_points_test)

            # Clean with algo if needed
            if apply_clean is True:
                s1_di_points = gp.clean_with_algo(
                    s1_di_points, algorithm='EllipticEnvelope')
                s2_di_points = gp.clean_with_algo(
                    s2_di_points, algorithm='EllipticEnvelope')

            # Train Classifier and get the trained classifier model
            classifier_model = classifier_train(np.append(s1_di_points, s2_di_points, axis=0), len(
                s1_di_points) * [0] + len(s2_di_points) * [1], classifier_algorithm, **kwargs)

            # Predict and get the predicted subject indeces from predicted labels of classifier
            predicted_subjects = [pair[0] if label == 0 else pair[1] for label in classifier_predict(
                ref_di_points_test, classifier_model)]

            # Update votes
            tpl = np.unique(predicted_subjects, return_counts=True)
            for j in range(len(tpl[0])):
                subjects_votes[tpl[0][j]] += tpl[1][j]

    # Return the final predicted subject index as the majority vote of the predicted subjects for each digraph
    return np.nanargmax(subjects_votes)


def experiment(subjects_extracted_list, classifier_algorithm, test_words, n_trials, apply_PCA=False, algo_clean=False, write_results_to_txt=True, dataset='MIXED', **kwargs):
    """Applies classification experiment to identify and predict keystroke patterns. \n
    Given a test_word it picks randomly a subject as a referrer. Then, it tries to 
    predict in which subject the test points belong with classification and One Vs One
    Reduction with Majority Vote.
    Parameters
    ----------
    `subjets_extracted_list` (list) The list of dicts with subjects data\n
    `classifier_algorithm` (string) The classifier algorithm\n
    `test_words` (list) The list of test_words for this experiment\n
    `n_trials` (int) The number of trials for this experiment\n
    `apply_PCA` (boolean) If you want to apply PCA before\n
    `algo_clean` (boolean) If you want to clean noise from data prior to classification\n
    `write_results_to_txt` (boolean) If you want to write results to txt\n
    `dataset` (str) Specify which dataset is used\n

    `kwargs` Can be knn_neighbors, 
    Returns
    ----------
    None
    """
    print 'Classification OneVsOne Experiment started.'
    start = time.time()

    # Fix parameters
    if not isinstance(subjects_extracted_list, list):
        subjects_extracted_list = [subjects_extracted_list]
    if not isinstance(test_words, list):
        test_words = [test_words]

    # Begin Experiment
    ERRORS = 0.
    count_global = 0.

    # Perform many trials
    for trial in range(n_trials):

        # Test each word
        for word in test_words:

            # Pick randomly a subject as referrer
            referrer_index = randint(0, len(subjects_extracted_list) - 1)

            # Test the word
            prediction_index = word_test(
                subjects_extracted_list, referrer_index, word, classifier_algorithm, apply_PCA,  apply_clean=algo_clean, kwargs=kwargs)
            count_global += 1

            # Is the prediction wrong?
            if prediction_index != referrer_index:
                ERRORS += 1

    # Calculate Average Pred Error
    ERRORS = 100 * ERRORS / count_global

    # Write results
    res_str = ''
    res_str += '\n-----------------'
    now = str(datetime.datetime.now())[:-7]
    res_str += '# Date: ' + now + '----------'
    res_str += '\n# Subjects: ' + \
        str(len(subjects_extracted_list)) + ' (' + dataset + ')'
    res_str += '\n# Test Words' + str(test_words)
    res_str += '\n# Trials: ' + str(n_trials)
    res_str += '\n# Algorithm: ' + str(classifier_algorithm)
    res_str += '\n# Data PCA prior: ' + str(apply_PCA)
    res_str += '\n# Data Cleaned prior: ' + str(algo_clean)
    res_str += '\nPredictionError = %.3f %%' % (ERRORS)
    res_str += '\n-------Experiment finished in ' + \
        str(time.time() - start) + ' s.------------\n'

    if write_results_to_txt is True:
        with open('classification-maj-vote-results' + '.txt', 'a') as fin:
            fin.write(res_str)

    return ERRORS
