# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""Produces plots for various algorithm parameters to find
the best parameters that minimizes the Classification Prediction Error
"""
import classification_maj_vote
import read_write as rw
import extract
import general_purpose as gp
import numpy as np
import matplotlib.pyplot as plt
import time

WORD_SET = [' t']


def svm_err_over_gammas(sed_list, test_words, n_trials, gammas):
    """Plots Prediction Error across various values of SVM gamma parameter, for 3d and 2d.\n
    It will give you a plot with two lines, One for 3d and One for 2d with pca.
    """
    errors_dict = dict((g, {'3d': 0., '2d': 0.}) for g in gammas)
    for gamma in gammas:
        errors_dict[gamma]['3d'] = classification_maj_vote.experiment(
            sed_list, 'SVC', test_words, n_trials, apply_PCA=False, svc_gamma=gamma)
        errors_dict[gamma]['2d'] = classification_maj_vote.experiment(
            sed_list, 'SVC', test_words, n_trials, apply_PCA=True, svc_gamma=gamma)

    plt.semilogx(gammas, [errors_dict[g]['3d']
                          for g in gammas], label='3D', color='blue', linewidth='1.35')
    plt.semilogx(gammas, [errors_dict[g]['2d']
                          for g in gammas], label='2D with PCA', color='red', linewidth='1.35')
    plt.title('SVM Performance Against Gamma Parameter')
    plt.ylabel('Prediction Error %')
    plt.xlabel('Gamma')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def knn_err_over_neighbors(sed_list, test_words, n_trials, neighbors_list):
    """Plots Prediction Error across various values of kNN n_neighbors parameter"""
    errors_dict = dict((n, {'3d': 0., '2d': 0.}) for n in neighbors_list)
    for n_neighbors in neighbors_list:
        errors_dict[n_neighbors]['3d'] = classification_maj_vote.experiment(
            sed_list, 'KNeighborsClassifier', test_words, n_trials, apply_PCA=False, knn_neighbors=n_neighbors)
        errors_dict[n_neighbors]['2d'] = classification_maj_vote.experiment(
            sed_list, 'KNeighborsClassifier', test_words, n_trials, apply_PCA=True, knn_neighbors=n_neighbors)

    plt.plot(neighbors_list, [errors_dict[n]['3d']
                              for n in neighbors_list], label='3D', color='blue', linewidth='1.35')
    plt.plot(neighbors_list, [errors_dict[n]['2d']
                              for n in neighbors_list], label='2D with PCA', color='red', linewidth='1.35')
    plt.title('kNN Performance Against Number of Neighbors')
    plt.ylabel('Prediction Error %')
    plt.xlabel('Neighbors')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def main():
    """Data are read from the local file.
    """
    # Read Extracted timings from local
    sed_list = rw.read_timings_from_local('subjects-data.json')

    # Pick dataset to experiment.
    _l = gp.pick_from_list(gp.choose_from(
        sed_list, 'SAFESHOP'), [1, 2, 4, 6, 8, 9])

    # 1) Plot prediction Error of SVM over the gamma parameter for various values
    # svm_err_over_gammas(_l, WORD_SET, n_trials=1000,
    #                     gammas=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.8, 1, 3, 5, 10, 50])

    # 2) Plot Prediction Error of kNN over the n_neighbors parameter
    knn_err_over_neighbors(_l, WORD_SET, n_trials=1000,
                           neighbors_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


# Entry point
if __name__ == '__main__':
    main()
