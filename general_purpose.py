# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0330, C0303
"""General Purpose Functions"""
import numpy as np
from random import randint
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TRACK_CODE = {"RAT": '5537343160336', "SAFESHOP": '84288924810000'}


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


def find_digraph_many_samples(sed_list, min_samples=20):
    """Finds digraph with many samples (>=min_samples) from the given list of subjects.
    Parameters
    ----------
    \n `sed` (list) The subjects extracted timings list
    \n `min_samples` (int) How many samples, default `20`
    Returns
    ----------
    \n A list of digraphs. If nothing is found, the program `exits`.
    """
    sd = []
    for sed in sed_list:
        sd.append([{"digraph": d['digraph'], "samples":len(d['points'])}
                   for d in sed['data']])
    rets = []
    ref = sd[0]
    for di in ref:
        if di['samples'] < min_samples:
            continue
        else:
            if len(sed_list) == 1:
                rets.append(di['digraph'])
            else:
                di_to_search = di['digraph']
                for j, s in enumerate(sd[1:]):
                    _len = [d['samples'] for d in s if d['samples'] >=
                            min_samples and d['digraph'] == di_to_search]
                    if _len == []:
                        break
                    else:
                        if j == len(sd) - 2:
                            rets.append(di_to_search)
    if rets == []:
        exit('****Exiting no digraph with samples >= ' +
             str(min_samples) + ' for the given list of subjects.')
    return rets


def choose_from(sed_list, _from):
    """Choose subjects from `RAT` or `SAFESHOP`."""
    if _from != 'RAT' and _from != 'SAFESHOP':
        exit('Wrong _from argument')
    return [sed for sed in sed_list if sed['_track_code'] == TRACK_CODE[_from]]


def pick_from_list(_list, indeces):
    """Picks from list with indeces"""
    if indeces == []:
        return _list
    ret = []
    for i in indeces:
        ret.append(_list[i])
    return ret


def is_not_extreme_outlier(x, _min, _max):
    """Returns `True` if `x >= min` and `x <= max`."""
    return x >= _min and x <= _max


def my_reshape(sel, filter_by_digraph=''):
    """Reshapes the subject extracted data list.
    Parameters
    ----------
    \n `sel`(list) The subjects extracted timings list
    \n `filter_by_digraph` (string) Specifies the digraph to filter
    Returns
    ----------
    `Object` with keys ['subject'], ['track_code'], and ['points'] as x,y,z
    """
    if filter_by_digraph != '':
        tmp = [v for v in sel['data'] if v['digraph'] == filter_by_digraph]
        if tmp == []:
            # exit('!!!Exiting: No digraph data found for subject:' +
            #      sel['_subject'])
            return -1
        else:
            pts = tmp[0]['points']
    else:
        pts = sel['data'][0]['points']
        for v in sel['data'][1:]:
            pts = np.append(pts, v['points'], axis=0)
    return {"subject": sel['_subject'], "track_code": sel['_track_code'], "points": pts}


def is_inside_interval(point, m, t):
    """Returns: true if `point` is >= `m-t` and <= `m+t`"""
    return point >= m - t and point <= m + t


def clean_with_std(points, n_stds_tolerance):
    """Removes data that are too far away by n_stds of their mean.
    Parameters
    ----------
    \n`points` (np.array) x,y,z points
    \n`n_stds_tolerance` (int) How many stds tolerance
    Returns
    ---------
    \nA numpy array with clean data
    """
    means = {"x": np.mean(points[:, 0]), "y": np.mean(
        points[:, 1]), "z": np.mean(points[:, 2])}
    tols = {"x": n_stds_tolerance * np.std(points[:, 0]), "y": n_stds_tolerance * np.std(
        points[:, 1]), "z": n_stds_tolerance * np.std(points[:, 2])}
    return np.array([row for row in points if is_inside_interval(row[0], means['x'], tols['x']) and is_inside_interval(row[1], means['y'], tols['y']) and is_inside_interval(row[2], means['z'], tols['z'])])


def clean_with_algo(X, algorithm, contamination=0.1, visualize_results=False):
    """Applies the specific algorithm to remove outliers from the data, something like outlier
       detection to remove noise from data coming from one class.
       Parameters
       ----------
       \n `X` (np.array) NxM numpy array
       \n `algorithm` (string) Can be one of 'EllipticEnvelope' 
       \n `contamination` (float) If EllipticEnvelope  the contamination
       specifies how polluted the data are\n
       \n `visualize_results` (boolean) If you want to visualize results\n
       Returns
       ----------
       \nData without outliers, same shape as X
    """
    # Generate Model
    if hasattr(globals()[algorithm](), 'contamination'):
        model = globals()[algorithm](contamination=contamination)
    else:
        model = globals()[algorithm]()

    # Fit & predict
    model.fit(X)
    labels_pred = model.predict(X)

    # Remove outliers
    _X = np.array([row for i, row in enumerate(X) if labels_pred[i] != -1])

    # Visualize results
    if visualize_results is True:
        figure = plt.figure(figsize=plt.figaspect(2.))
        ax = Axes3D(figure)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=80,
                   c=['r' if l == 1 else 'k' for l in labels_pred], marker='o', alpha=0.9)
        plt.title('Noise removal with ' + algorithm)
        # plt.show()
    return _X


def events_sample(events, samples):
    """Samples a continuous amount of events defined by samples """
    if samples > len(events):
        exit('*events_sample: Exiting -> sample > events length')
    start = randint(0, len(events) - samples - 1)
    return events[start:start + samples]
