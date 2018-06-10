# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""This module is used to experiment with anomaly detection for various parameters of the algoritmhs."""
import anomaly
import read_write as rw
import extract
import general_purpose as gp
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

WORD_SET1 = ['in the']
# WORD_SET1 = ['in']

COLORS = ['b', 'r', 'g', 'm', 'c', 'k']
MARKERS = ['x', 'v', '+', '*', '>', '<']


def nanargmin_rev(seq):
    """Calculates minimum index of sequence, beginning from end."""
    return len(seq) - np.nanargmin(list(reversed(seq))) - 1


def svm_plot_eer_for_gammas_given_thresh(sed_list, gammas, threshold, trials, dimensions='both'):
    """Plots EER for various gammas given a fixed threshold for each dimensions you specify"""
    start = time.time()
    dims = [dimensions] if dimensions != 'both' else ['3-D', '2-D']
    for i, dim in enumerate(dims):
        EER = []
        for gamma in gammas:
            far, frr = anomaly.experiment(sed_list, test_words=WORD_SET1,
                                          trials=trials,
                                          testing_threshold=threshold,
                                          apply_PCA=(dim == '2-D'),
                                          algorithm='OneClassSVM',
                                          algo_params={"gamma": gamma},
                                          clean_referrer_with_algo=False,
                                          print_results=False, write_to_txt=False)
            EER.append(100 * (far + frr) / 2.)
        # Plot EER
        plt.semilogx(gammas, EER, linestyle='-',
                     color=COLORS[i], marker=MARKERS[i],
                     linewidth=1.5, alpha=1, label=dim)
        # Save results to txt
        with open('anomaly-oneclasssvm-gamma-performance' + '.txt', 'a') as fout:
            resstr = ''
            if i == 0:
                resstr += '\n\n----------------'
                now = str(datetime.datetime.now())[:-7]
                resstr += now + '--------'
                resstr += '\n# Threshold = ' + str(threshold)
            resstr += '\n# PCA = ' + str(dim == '2-D')
            resstr += '\n# Gammas = ' + str(gammas)
            resstr += '\nEER: ' + str(EER)
            fout.write(resstr)
    plt.title('OneClass SVM Gamma Performance')
    plt.legend(loc='upper right', framealpha=1, edgecolor='black')
    plt.xlabel('Gamma')
    plt.ylabel('EER %')
    print 'Evaluation finished in %.2fs.' % (time.time() - start)
    plt.show()


def svm_plot_farfrr_for_thresholds_given_gamma(sed_list, gamma, thresholds, trials, dimensions='both'):
    """Plots FAR,FRR for given gamma across thresholds"""
    print 'Anomaly Evaluation Started.'
    start = time.time()

    # Initialize Stuff
    dims = [dimensions] if dimensions != 'both' else ['3-D', '2-D']
    FAR, FRR = dict((d, []) for d in dims), dict((d, []) for d in dims)

    # Begin
    for d, dim in enumerate(dims):
        for thresh in thresholds:
            far, frr = anomaly.experiment(sed_list, WORD_SET1, trials=trials,
                                          testing_threshold=thresh,
                                          apply_PCA=(dim == '2-D'),
                                          algorithm='OneClassSVM',
                                          algo_params={"gamma": gamma[dim]},
                                          print_results=False,
                                          write_to_txt=False)
            FAR[dim].append(100 * far)
            FRR[dim].append(100 * frr)

        # Plot FAR and FRR for given dimension
        plt.plot(thresholds, FAR[dim], linestyle='-',
                 marker=MARKERS[d], color=COLORS[d],
                 label='FAR - ' + str(dim),
                 linewidth=1.0, alpha=0.94, zorder=1)
        plt.plot(thresholds, FRR[dim], linestyle='--',
                 marker=MARKERS[d], color=COLORS[d],
                 label='FRR - ' + str(dim),
                 linewidth=1.0, alpha=0.94, zorder=1)

        # Find EER point and add it to plot
        eer_ind = nanargmin_rev(
            np.abs(np.array(FAR[dim]) - np.array(FRR[dim])))
        EER = {"x": thresholds[eer_ind], "y": (
            FAR[dim][eer_ind] + FRR[dim][eer_ind]) / 2.}
        plt.plot(EER['x'], EER['y'], 'ko', markersize=6, alpha=1, zorder=2)

        # Save the points of plot
        with open('anomaly-oneclasssvm-threshold-performance' + '.txt', 'a') as fout:
            resstr = ''
            if d == 0:
                resstr += '\n\n----------------'
                now = str(datetime.datetime.now())[:-7]
                resstr += now + '-----------------'
                resstr += '\n# Thresholds = ' + str(thresholds)
            resstr += '\n# PCA = ' + \
                str(dim == '2-D') + ', Gamma = ' + str(gamma[dim])
            resstr += '\nFAR: ' + str(FAR[dim])
            resstr += '\nFRR: ' + str(FRR[dim])
            resstr += '\nEER: ' + str(EER)
            fout.write(resstr)

    # Configure Plot and show it.
    plt.title('One-Class SVM Performance')
    plt.legend(loc='upper right', framealpha=1, edgecolor='black')
    plt.xlabel('Threshold')
    plt.ylabel('Error %')
    plt.yticks(np.arange(0, 40, 5.0))
    plt.ylim(0, 40)
    # plt.xticks(np.arange(min(thresholds), max(thresholds), 0.05))
    print 'Evaluation finished in %.2fs.' % (time.time() - start)
    plt.show()


def gmm_plot_eer_for_deltas_and_ncomps_given_thresh(sed_list, deltas, n_components, threshold, trials, with_pca):
    """Plots EER for various deltas and n_components for given threshold"""
    print 'Anomaly Evaluation Started.'
    start = time.time()

    # Init
    if not isinstance(n_components, list):
        n_components = [n_components]
    EER = dict((n, []) for n in n_components)

    # Begin
    for i, n_comps in enumerate(n_components):
        for delta in deltas:
            far, frr = anomaly.experiment(sed_list, test_words=WORD_SET1,
                                          trials=trials,
                                          testing_threshold=threshold,
                                          apply_PCA=with_pca,
                                          algorithm='GMM',
                                          algo_params={
                                              "n_components": n_comps, "delta": delta},
                                          clean_referrer_with_algo=False,
                                          print_results=False, write_to_txt=False)
            EER[n_comps].append(100 * (far + frr) / 2.)

        # Plot EER
        plt.plot(deltas, EER[n_comps], linestyle='-',
                 marker=MARKERS[i], color=COLORS[i], label='%dG' % (n_comps), linewidth=1.5, alpha=0.9)

        # Save results to txt
        with open('anomaly-gmm-delta-performance' + '.txt', 'a') as fout:
            resstr = ''
            if i == 0:
                resstr += '\n\n----------------'
                now = str(datetime.datetime.now())[:-7]
                resstr += now + '-----------------'
                resstr += '\n# Threshold = ' + str(threshold)
                resstr += '\n# PCA = ' + str(with_pca)
            resstr += '\n# Components = ' + str(n_comps)
            resstr += '\n# Deltas =  ' + str(deltas)
            resstr += '\nEER: ' + str(EER[n_comps])
            fout.write(resstr)

    # Configure plot and show it
    plt.title('GMM Delta Performance ' +
              ('(3-D)' if with_pca is False else '(2-D)'))
    plt.legend(loc='upper right', framealpha=1, edgecolor='black')
    plt.ylim(0, 30)
    plt.xlabel('Distance $\delta$')
    plt.ylabel('EER %')
    print 'Evaluation finished in %.2fs.' % (time.time() - start)
    plt.show()


def gmm_plot_farfrr_for_thresholds_and_ncomps_given_delta(sed_list, n_components_list, thresholds, deltas, trials, with_pca=False):
    """Plots FAR and FRR for each GMM with ncomps, for a given DELTA for each comp across the thresholds axis.\n
    It will give you one plot with 2*n lines where n = len(n_components_list)-> one pair of FAR-FRR lines for each component\n
    Parameters
    ----------
    `sed_list` (list) A list with subjects\n
    `n_components_list` (list) A list with how many components each GMM will have\n
    `DELTAS` (dict) A dict with the delta for each component\n
    `trials` (int) How many trials to perform
    """
    print 'Anomaly Evaluation Started.'
    start = time.time()

    # Init
    FAR, FRR = dict((n, []) for n in n_components_list), dict((n, [])
                                                              for n in n_components_list)
    # Begin
    for i, ncomp in enumerate(n_components_list):
        for thresh in thresholds:
            far, frr = anomaly.experiment(sed_list, WORD_SET1,
                                          trials=trials, testing_threshold=thresh,
                                          algorithm='GMM',
                                          algo_params={
                                              "n_components": ncomp,
                                              "delta": deltas[ncomp]},
                                          apply_PCA=with_pca,
                                          clean_referrer_with_algo=False,
                                          print_results=False, write_to_txt=False)
            FAR[ncomp].append(100 * far)
            FRR[ncomp].append(100 * frr)

        # Plot FAR and FRR for given gamma
        plt.plot(thresholds, FAR[ncomp], linestyle='-',
                 marker=MARKERS[i], color=COLORS[i], label=str(ncomp) + 'G-FAR, $\delta$: ' +
                 str(deltas[ncomp]),
                 linewidth=1.0, alpha=0.94, zorder=1)
        plt.plot(thresholds, FRR[ncomp], linestyle='--',
                 marker=MARKERS[i], color=COLORS[i], label=str(ncomp) + 'G-FRR, $\delta$: ' +
                 str(deltas[ncomp]),
                 linewidth=1.0, alpha=0.94, zorder=1)

        # Find EER point and add it to plot
        eer_ind = nanargmin_rev(
            np.abs(np.array(FAR[ncomp]) - np.array(FRR[ncomp])))
        EER = {"x": thresholds[eer_ind],
               "y": (FAR[ncomp][eer_ind] + FRR[ncomp][eer_ind]) / 2.,
               "FAR,FRR": str(FAR[ncomp][eer_ind]) + ', ' + str(FRR[ncomp][eer_ind])}
        plt.plot(EER['x'], EER['y'], 'ko', markersize=6, alpha=1, zorder=2)
        print 'EER for N_Components =' + str(ncomp)
        print EER

        # Save Data to txt
        with open('anomaly-gmm-threshold-performance' + '.txt', 'a') as fout:
            mystr = ''
            if i == 0:
                mystr += '\n\n----------------'
                now = str(datetime.datetime.now())[:-7]
                mystr += now + '-----------------'
                mystr += '\n## Subjects: ' + str(len(sed_list))
                mystr += '\n## Thresholds: ' + str(thresholds)
                mystr += '\n## PCA: ' + str(with_pca)
            mystr += '\n# Components: ' + str(ncomp)
            mystr += '\n# Delta: ' + str(deltas[ncomp])
            mystr += '\n FAR: ' + str(FAR[ncomp])
            mystr += '\n FRR: ' + str(FRR[ncomp])
            mystr += '\n EER: ' + str(EER)
            if i == len(n_components_list) - 1:
                mystr += '\n-------------------------------\n\n'
            fout.write(mystr)

    # Configure plot and show it
    plt.title('GMM Performance ' +
              ('(3-D)' if with_pca is False else '(2-D)'))
    plt.legend(loc='upper right', framealpha=1, edgecolor='black')
    plt.xlabel('Threshold')
    plt.ylabel('Error %')
    plt.yticks(np.arange(0, 31, 5.0))
    plt.ylim(0, 30)
    # plt.xticks(np.arange(min(thresholds), max(thresholds), 0.1))
    print 'Evaluation finished in %.2fs.' % (time.time() - start)
    plt.show()


def main():
    """
    """
    # Read Extracted timings from local
    sed_list = rw.read_timings_from_local('subjects-data.json')

    # Pick dataset to experiment.
    _l = gp.pick_from_list(gp.choose_from(
        sed_list, 'SAFESHOP'), [1, 6, 8, 9])

    # Plot eer for various parameters and algos
    # ------ One Class SVM stuff ---------
    # svm_plot_eer_for_gammas_given_thresh(
    #     _l, gammas=[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025,
    #                 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10],
    #     threshold=0.4,
    #     trials=100,
    #     dimensions='both')
    svm_plot_farfrr_for_thresholds_given_gamma(
        _l, gamma={
            '2-D': 1},
        thresholds=np.arange(0.1, 0.81, 0.01),
        trials=60, dimensions='2-D')

    # ------ GMM Stuff --------
    # gmm_plot_eer_for_deltas_and_ncomps_given_thresh(_l,
    #                                                 deltas=np.arange(
    #                                                     0.5, 3.1, 0.1),
    #                                                 n_components=[1, 2, 3],
    #                                                 threshold=0.3, trials=80,
    #                                                 with_pca=True)
    # gmm_plot_farfrr_for_thresholds_and_ncomps_given_delta(_l,
    #                                                       n_components_list=[
    #                                                           1, 2, 3],
    #                                                       thresholds=np.arange(
    #                                                           0.1, 0.81, 0.01),
    #                                                       deltas={1: 0.9,
    #                                                               2: 1.4,
    #                                                               3: 1.9},
    #                                                       trials=60, with_pca=True)



# Entry point
if __name__ == '__main__':
    main()


# def gmm_plot_1_experiment(sed_list, deltas, thresholds, n_components_list, trials, with_pca=False):
#     """It will give you a figure with n subplots, one for each number of component.\n
#     It plots the EER for various values of thresholds over the x axis of deltas.
#     So for each subplot you have t lines where t = len(thresholds).
#     Parameters
#     --------
#     `sed_list` The list with subjects\n
#     `deltas` A list with values for delta\n
#     `thresholds` A dict in this form: {n: [t1,t2,t3],..,} where n the value of component
#     and t1,t2,t3 the values of threshold that will be examined for that component\n
#     `n_components_list` A list with n_components\n
#     `trials` The number of trials
#     """
#     print 'Anomaly Evaluation Started.'
#     start = time.time()
#     if not isinstance(n_components_list, list):
#         n_components_list = [n_components_list]
#     # Init Figure: n Subplots, One for each Component
#     fig = plt.figure()
#     figdim1, figdim2 = 1, len(n_components_list)
#     for i, n_comp in enumerate(n_components_list):

#         # Init subplot
#         ax = fig.add_subplot(figdim1, figdim2, i + 1)

#         # Calculate EER for various thres and deltas
#         EER = dict((t, []) for t in thresholds[n_comp])
#         for j, threshold in enumerate(thresholds[n_comp]):
#             for delta in deltas:
#                 far, frr = anomaly.experiment(sed_list, WORD_SET1, trials=trials, testing_threshold=threshold,
#                                               algorithm='GMM', apply_PCA=with_pca,
#                                               algo_params={
#                                                   "n_components": n_comp, "delta": delta},
#                                               print_results=False, write_to_txt=False)
#                 EER[threshold].append(100 * (far + frr) / 2.)

#             # print '%d Components,' % (n_comp)
#             # print 'Best EER For Threshold: ' + str(threshold)
#             # print '-> ' + str(np.min(EER[threshold])) + 'in Delta = ' + str(deltas[np.nanargmin(EER[threshold])])
#             # print '\n'

#             # Plot EER
#             ax.plot(deltas, EER[threshold], linestyle='-',
#                     marker=MARKERS[j], color=COLORS[j], label='Threshold = ' + str(threshold), linewidth=1.2, alpha=0.9)
#             ax.set_title('%dG' % (n_comp))
#             ax.legend(loc='upper right')
#             ax.set_xlabel('Distance $\delta$')
#             ax.set_ylabel('EER %')

#         # Save EER to txt (you never know)
#         with open('gmm-data-exp1' + '.txt', 'a') as fout:
#             mystr = ''
#             if i == 0:
#                 mystr += 'PCA: ' + str(with_pca) + '\n'
#             mystr = '# Components: ' + str(n_comp)
#             mystr += '\n# Thresholds: ' + str(thresholds[n_comp])
#             mystr += '\n# Deltas: ' + str(deltas)
#             mystr += '\n EER: ' + str(EER) + '\n\n'
#             if i == len(n_components_list) - 1:
#                 mystr += '\n-------------------------------\n'
#             fout.write(mystr)

#     print 'Evaluation finished in %.2fs.' % (time.time() - start)

#     suptitle = 'GMM Performance'
#     if with_pca is True:
#         suptitle += ',with PCA-2D'
#     plt.suptitle('GMM Performance')
#     plt.show()
