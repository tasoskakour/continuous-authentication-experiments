"""This file is only for some plots. Dont take it seriously"""

# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0330
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = ['darkred', 'red', '#f66767']  # third is lightred


def compute_malahanobis(mu, S, x):
    """Computes malahanobis distance of point x(1xD, D dimensions) from the distribution with means and covs"""
    mlh = np.sqrt(
        np.matmul(np.matmul((x - mu), np.linalg.inv(S)), np.transpose(x - mu)))
    return mlh


def gmm_make_contours(gmm, ax):
    CMAP = ['binary', plt.cm.PuBu_r]
    xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                         np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
    zz = np.c_[xx.ravel(), yy.ravel()]

    mlh_contour = 0.
    for c in range(len(gmm.weights_)):
        mlh = []
        for z in zz:
            mlh.append(compute_malahanobis(
                gmm.means_[c], gmm.covariances_[c], z))
        mlh = np.array(mlh).reshape(xx.shape)
        mlh_contour = ax.contour(xx, yy, mlh,
                                 cmap=CMAP[c],
                                 linestyles='dashed')
    return mlh_contour.collections[1]


def make_GMM_ellipses(gmm, ax):
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


X_train = np.array([
    [127.0, 140.0, -28.0],
    [134.0, 158.0, 4.0],
    [128.0, 135.0, 50.0],
    [99.0, 140.0, -32.0],
    [120.0, 170.0, 74.0],
    [135.0, 181.0, -49.0],
    [109.0, 145.0, 13.0],
    [131.0, 135.0, -62.0],
    [172.0, 142.0, -78.0],
    [117.0, 152.0, -84.0],
    [122.0, 157.0, 30.0],
    [129.0, 140.0, -40.0],
    [87.0, 153.0, 66.0],
    [135.0, 161.0, -89.0],
    [128.0, 215.0, -42.0],
    [130.0, 171.0, -7.0],
    [119.0, 145.0, -52.0],
    [134.0, 120.0, -38.0],
    [125.0, 139.0, 9.0],
    [124.0, 91.0, -57.0],
    [131.0, 160.0, -69.0],
    [142.0, 137.0, -46.0],
    [127.0, 168.0, -86.0],
    [144.0, 117.0, -73.0],
    [138.0, 146.0, -71.0],
    [131.0, 154.0, -64.0],
    [129.0, 117.0, -62.0],
    [128.0, 149.0, -51.0],
    [107.0, 157.0, -42.0],
    [109.0, 178.0, 49.0],
    [107.0, 141.0, -12.0],
    [104.0, 162.0, 10.0],
    [53.0, 123.0, 8.0],
    [128.0, 88.0, 52.0],
    [109.0, 186.0, -9.0],
    [132.0, 176.0, -32.0],
    [125.0, 117.0, -46.0],
    [133.0, 171.0, -28.0],
    [137.0, 139.0, -72.0],
    [133.0, 158.0, -74.0],
    [128.0, 171.0, -79.0],
    [134.0, 131.0, -71.0],
    [119.0, 112.0, -46.0],
    [152.0, 152.0, -88.0],
    [114.0, 183.0, -13.0],
    [128.0, 135.0, -81.0],
    [117.0, 155.0, -11.0],
    [118.0, 165.0, -48.0],
    [134.0, 156.0, -26.0],
    [119.0, 167.0, -45.0],
    [107.0, 163.0, -61.0],
    [116.0, 212.0, 28.0],
    [139.0, 157.0, -28.0],
    [132.0, 181.0, 12.0],
    [146.0, 155.0, -93.0],
    [152.0, 185.0, -47.0],
    [139.0, 160.0, -61.0],
    [141.0, 134.0, -63.0],
    [142.0, 131.0, -38.0],
    [131.0, 147.0, -83.0],
    [104.0, 147.0, 31.0],
    [127.0, 144.0, -67.0],
    [88.0, 158.0, -57.0],
    [119.0, 133.0, -4.0],
    [124.0, 118.0, -77.0],
    [128.0, 167.0, -63.0],
    [138.0, 164.0, -80.0],
    [116.0, 143.0, -49.0],
    [131.0, 151.0, -25.0],
    [146.0, 182.0, -78.0],
    [113.0, 99.0, -7.0],
    [126.0, 178.0, -109.0],
    [105.0, 193.0, -4.0],
    [90.0, 162.0, 13.0],
    [127.0, 148.0, -24.0],
    [112.0, 131.0, -65.0],
    [138.0, 139.0, -41.0],
    [143.0, 154.0, -64.0],
    [124.0, 129.0, -15.0],
    [136.0, 146.0, -30.0],
    [124.0, 150.0, -49.0],
    [121.0, 172.0, -25.0],
    [136.0, 142.0, -51.0],
    [140.0, 130.0, -79.0],
    [148.0, 148.0, -91.0],
    [111.0, 150.0, -38.0],
    [142.0, 105.0, 24.0],
    [152.0, 147.0, -98.0],
    [130.0, 142.0, -55.0],
    [180.0, 163.0, -90.0],
    [133.0, 146.0, -74.0],
    [128.0, 144.0, -20.0],
    [118.0, 155.0, -30.0],
    [126.0, 248.0, -13.0],
    [140.0, 123.0, -64.0],
    [125.0, 138.0, 11.0],
    [127.0, 150.0, -4.0],
    [106.0, 134.0, -36.0],
    [143.0, 129.0, -72.0],
    [151.0, 154.0, -15.0],
    [129.0, 147.0, -24.0],
    [132.0, 132.0, -24.0],
    [108.0, 127.0, -10.0],
    [96.0, 100.0, -78.0],
    [124.0, 164.0, -38.0]
])

X_test = np.array([[92.0, 101.0, 303.0],
                   [64.0, 82.0, 121.0],
                   [100.0, 105.0, -14.0],
                   [100.0, 92.0, -25.0]])

merged = np.append(X_train, X_test, axis=0)
# Scale
sscaler = StandardScaler(with_mean=True, with_std=True).fit(merged)
X_train = sscaler.transform(X_train)
X_test = sscaler.transform(X_test)
merged = np.append(X_train, X_test, axis=0)

# Reduce
pca = PCA(n_components=2).fit(merged)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# Fit GMM
gmm = GaussianMixture(n_components=2, covariance_type='full').fit(X_train)

# Init figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Plot train
ax_train = ax.scatter(X_train[:, 0], X_train[:, 1], color='red',
                      s=80, alpha=1)
# Plot gaussian components
make_GMM_ellipses(gmm, ax)

# Plot test
ax_test = ax.scatter(X_test[:, 0], X_test[:, 1], color='blue',
                     s=80, alpha=1)

# Make mlh contour
# ax_contour = gmm_make_contours(gmm, ax)

# Compute malahanobis for test for each component
for c in range(len(gmm.weights_)):
    for x in X_test:
        print compute_malahanobis(gmm.means_[c], gmm.covariances_[c], x)
    print '\n'

# ax.legend([ax_train, ax_test, ax_contour], [
#           'Train', 'Test', 'MLH Distance'])
ax.legend([ax_train, ax_test], [
          'Train', 'Test'])
ax.set_title('Digraph: KeyHKeyE')
plt.show()
