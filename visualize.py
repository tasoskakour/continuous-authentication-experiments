# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0303
"""This module is used for visualization of data."""
import numpy as np
import general_purpose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA


S = 80
ALPHA = 0.8
MAIN_MARKER = 'o'
MARKERS = ['o', 'x', 'v', '^', '>', '<', '1', '2']
COLORS_SET_1 = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'pink']
COLORS_SET_2 = ['g', 'c', 'm', 'r', 'm', 'b', 'k', 'y']


def add_scatter_subplot(figure, subjects_table, subplot_count, subplot_title, subplot_dim_1, subplot_dim_2, labels_pred=[], _3d=True,  xlabel='Key_1 Hold(ms)',  ylabel='Key_2 Hold(ms)', zlabel='Digraph Up_Down(ms)', show_legend=True):
    """"""

    # List of scatter pts
    plts = []

    # Create sub-plot
    if _3d is True:
        ax = figure.add_subplot(subplot_dim_1, subplot_dim_2,
                                subplot_count, projection='3d')
    else:
        ax = figure.add_subplot(subplot_dim_1, subplot_dim_2, subplot_count)

    # Check if you have labels, otherwise just plot them
    if labels_pred == []:

        # Apply points to sub-plot
        for i, d in enumerate(subjects_table):
            if _3d is True:
                plts.append(ax.scatter(d['points'][:, 0], d['points'][:, 1],
                                       d['points'][:, 2], s=S, marker=MAIN_MARKER, color=COLORS_SET_1[i], alpha=ALPHA))
            else:
                plts.append(ax.scatter(d['points'][:, 0], d['points'][:, 1],
                                       s=S, marker=MAIN_MARKER, color=COLORS_SET_1[i], alpha=ALPHA))

        # Apply legend
        if show_legend is True:
            ax.legend(tuple(plts), tuple(
                [d['subject'] + ', n = ' + str(len(d['points'])) for d in subjects_table]), loc=0, bbox_to_anchor=(0, 0, 1, 0.945))
        else:
            ax.axis('off')
    else:

        # Determine predicted clusters from labels
        points_merged = subjects_table[0]['points']
        points_merged = np.append(
            points_merged, [row for s in subjects_table[1:] for row in s['points']], axis=0)
        s_predicted = len(subjects_table) * [None]
        for i, val in enumerate(points_merged):
            ind = labels_pred[i]
            if s_predicted[ind] is None:
                s_predicted[ind] = np.array([val])
            else:
                s_predicted[ind] = np.append(s_predicted[ind], [val], axis=0)

        # Apply points to sub-plot
        for i, pts in enumerate(s_predicted):
            if _3d is True:
                plts.append(ax.scatter(pts[:, 0], pts[:, 1],
                                       pts[:, 2], s=S, marker=MAIN_MARKER, color=COLORS_SET_2[i], alpha=ALPHA))
            else:
                plts.append(ax.scatter(
                    pts[:, 0], pts[:, 1], s=S, marker=MAIN_MARKER, color=COLORS_SET_2[i], alpha=ALPHA))

    # Apply  title and axis labels
    ax.set_title(subplot_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if _3d is True:
        ax.set_zlabel(zlabel)


def my_scatter(subjects_extracted_list,  digraphs=[''], standard_scaler=False, apply_PCA=False, algo_clean_flag=False, algo_clean_name='EllipticEnvelope', algo_clean_visualize_results=False, algo_clean_contamination=0.1):
    """Scatters in the same plot the data of  subjects_extracted_list
       TODO: different variations 2d, 1d etc
    \nParameters:
    \nsubjects_extracted_list: The list with the timings of all subjects that you want to plot
    \nkind_of: 'digraph_up_down'  or 'key_hold' or 'digraph_all'
    \nby_specific: If kind_of = 'key_hold' then if you can specify which keyCode to plot
    for those subjects timings. If you leave it '' then it will plot them all together.
    \n If kind_of !='key_hold' then it specfies the digraph in the same manner"""

    # Fix parameters
    if not isinstance(subjects_extracted_list, list):
        subjects_extracted_list = [subjects_extracted_list]
    if not isinstance(digraphs, list):
        digraphs = [digraphs]

    # Initialize Figure
    figure = plt.figure()

    # Calculate dimensions that each sub-plot will have
    subplot_dim_1 = np.ceil(np.sqrt(len(digraphs)))
    subplot_dim_2 = np.ceil(np.sqrt(len(digraphs)))

    # For each digraph
    for i, digraph in enumerate(digraphs):

        subjects_table = []
        for se in subjects_extracted_list:

            # Construct subjects_table with points
            tmp = general_purpose.my_reshape(se, digraph)
            if tmp != -1:
                subjects_table.append(general_purpose.my_reshape(se, digraph))
            else:
                print '***Warning: No samples found for digraph "' + digraph + '" of subject "' + se['_subject'] + '"'
                continue

        # Title and labels stuff
        subplot_title = 'All Digraphs' if digraph == '' else digraph
        xlabel = 'Key_1 Hold (ms)'
        ylabel = 'Key_2 Hold (ms)'
        zlabel = 'Digraph Up-Down (ms)'

        # Scale if needed
        if standard_scaler is True:
            my_scaler = StandardScaler(with_mean=True, with_std=True).fit(
                np.array([row for s in subjects_table for row in s['points']]))
            for s in subjects_table:
                s['points'] = my_scaler.transform(s['points'])
            subplot_title += ', +Scaled'
            xlabel = 'Key_1 Hold'
            ylabel = 'Key_2 Hold'
            zlabel = 'Digraph Up-Down'

        # Apply PCA if needed
        if apply_PCA is True:
            my_pca = PCA().fit(
                np.array([row for s in subjects_table for row in s['points']]))
            for s in subjects_table:
                s['points'] = my_pca.transform(
                    s['points'])[:, 0:2]  # keep first two dimensions
            # print my_pca.explained_variance_ratio_
            pca_info_loss = my_pca.explained_variance_ratio_[2]
            subplot_title += ', +PCA (Loss: %.0f%%)' % (100 * pca_info_loss)
            xlabel = 'u'
            ylabel = 'v'

        # Clean with algo if needed
        if algo_clean_flag is True:
            for s in subjects_table:
                s['points'] = general_purpose.clean_with_algo(
                    s['points'], algorithm=algo_clean_name, contamination=algo_clean_contamination, visualize_results=algo_clean_visualize_results)
            subplot_title += ', +Cleaned'

        # If the sub-plots are too many, don't show labes, titles etc

        show_legend = True
        if subplot_dim_1 > 3 and subplot_dim_2 > 3:
            xlabel, ylabel, zlabel = '', '', ''
            show_legend = False
            subplot_title = 'All Digraphs\n' if digraph == '' else digraph + '\n'

        # Add the scatter subplot
        _3d = True if subjects_table[0]['points'].shape[1] == 3 else False
        add_scatter_subplot(figure, subjects_table,
                            i + 1, subplot_title, subplot_dim_1, subplot_dim_2, _3d=_3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show_legend=show_legend)

    # Show the plots
    if subjects_table != []:
        if subplot_dim_1 > 3 and subplot_dim_2 > 3:
            suptitle = ('Subjects: ' +
                        str([v['_subject'] for v in subjects_extracted_list]))
            if standard_scaler is True:
                suptitle += ', +Scaled'
            if apply_PCA is True:
                suptitle += ', +PCA'
            if algo_clean_flag is True:
                suptitle += ', +Cleaned'
            plt.suptitle(suptitle)
        plt.xtick()
        plt.show()
    else:
        print '\n----> Nothing to plot. No data...'
