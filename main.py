# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""This module is used as an entry point for experiments"""
import read_write as rw
import extract
import general_purpose as gp
import sys
import visualize
import classification
import classification_maj_vote
import anomaly

# Word sets for experiments
# WORD_SET1 = ['in the', 'in there', 'there',
#              'here', 'here to']
WORD_SET1 = ['in the']
WORD_SET2 = ['in', 'th', 'in the']


def main():
    """Arguments from terminal [1]: 'db' (to load events from database, and then extract timings)
       or 'local' (to load extracted timings immediately from local)
    """
    # Process args
    args = sys.argv[1:]
    if args == []:
        read_from = 'db'
    else:
        read_from = args[0]

    # LOAD FROM DB
    if read_from == 'db':

        # Load docs from database
        docs = rw.load_from_mongodb()

        # Write raw docs to local
        rw.write_docs_to_local(docs, filename='subjects-docs')

        # Extract Timings
        sed_list = extract.all(docs, write_to_json=True, ignore_space=False)

    # LOAD FROM LOCAL
    else:

        # Read from local
        sed_list = rw.read_timings_from_local('subjects-data.json')

    # Pick subjects to experiment.
    _l1 = gp.pick_from_list(gp.choose_from(
        sed_list, 'SAFESHOP'), [1, 2,  4, 5, 6, 8, 9])  # || 1, 2,  4, 6, 8, 9
    _l2 = gp.pick_from_list(gp.choose_from(
        sed_list, 'RAT'), [7, 8, 9, 10, 17,  19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
    _l = _l1 + _l2
    # _l = gp.pick_from_list(sed_list, [16, 16])

    # You can pick if you want only from digraphs with many samples for each subject specified in the below func call
    # digraphs_many_samples = gp.find_digraph_many_samples(_l, min_samples=20)
    # print digraphs_many_samples
    # exit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                           Experiments below here
    #                  (just comment out which experiment you don't want)

    # -------------------------------1) Visualize--------------------------------------['KeyTKeyH', 'KeyHKeyE']
    # visualize.my_scatter(
    #     _l, digraphs=['KeyESpace', 'KeyHKeyE', 'KeyTKeyH', 'SpaceKeyT'], standard_scaler=False, apply_PCA=False, algo_clean_flag=False)

    # -------------------------------2) Classification-----------------------------------

    # i) Just Normal Classification
    # classification.experiment(
    #     # ['KeyTKeyH', 'KeyIKeyN', 'SpaceKeyT', 'KeyESpace']
    #     # ['KeyESpace', 'KeyHKeyE', 'KeyTKeyH', 'KeyTSpace', 'SpaceKeyO', 'SpaceKeyT'],
    #     _l1 + _l2, ['SVC', 'KNeighborsClassifier'], digraphs=['KeyIKeyN', 'KeyTKeyH'],
    #     n_trials=100,
    #     standard_scaler=True, apply_PCA=True,
    #     visualize_results=False, algo_clean=False, write_results_to_txt=True, dataset='MIXED')

    # ii) Classification with OneVsOne Reduction and Majority Vote
    # classification_maj_vote.experiment(_l2, test_words=['in'], n_trials=1000,  # WORD_SET1
    #                                    apply_PCA=False,
    #                                    classifier_algorithm='SVC', svc_gamma=0.25,
    #                                    dataset='SAFESHOP')

    # -------------------------------3) Anomaly Detection--------------------------------
    # (FAR, FRR) = anomaly.experiment(_l1, test_words=WORD_SET1,
    #                                 trials=100,
    #                                 testing_threshold=0.38,
    #                                 apply_PCA=False,
    #                                 visualize_results=False,
    #                                 algorithm='OneClassSVM', algo_params={"gamma": 0.5},
    #                                 dataset='SAFESHOP', clean_referrer_with_algo=False, write_to_txt=False)
    # (FAR, FRR) = anomaly.experiment(_l1, test_words=['in the'],
    #                                 trials=100,
    #                                 testing_threshold=0.29,
    #                                 apply_PCA=True,
    #                                 visualize_results=False,
    #                                 algorithm='GMM', algo_params={"n_components": 1, "delta": 0.9},
    #                                 clean_referrer_with_algo=False, write_to_txt=False, dataset='SAFESHOP')


# Entry point
if __name__ == '__main__':
    main()
