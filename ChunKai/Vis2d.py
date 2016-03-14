import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

"""
Heatmap and path plotter
"""


class Vis2d:
    def __init__(self):
        pass

    def MapPlot(self, grid_extent, ground_truth=None, posterior_mean_before=None, posterior_mean_after=None,
                posterior_variance_before=None, posterior_variance_after=None, path_points=None, display=True,
                save_path=None):
        """
        Plots original field and path taken, as well as
        Saves data to file if required
        @param ground_truth, posterior mean, posterior variance - 2d-array of relvant data.
        - Each 2d array represents 1 field (eg. posterior, ground truth)
        - Note that these are all indexed in integers etc
        - We will need to scale and translate accordingly
        @param grid_extent - axis mesh points as a 4-tuple of numpy arrays comprising (x-min, xmax, ymin, ymax)
        @param path_points - path coordinates in "world" space (ie. actual coordinates, not just (5,3) etc ... )
        @param display - shows plot on screen, useful for debugging
        @param save_path
        """

        grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                        grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

        mmax = -10 ** 10
        mmin = 10 ** 10
        for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
        #for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
            if not q == None:
                mmax = max(np.amax(np.amax(q)), mmax)
                mmin = min(np.amin(np.amin(q)), mmin)
        axes = plt.axes()
        #fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        if not ground_truth == None:
            im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent2,
                                     cmap='Greys', vmin=mmin, vmax=mmax)
            if not path_points == None and path_points:
                # batch size
                k = path_points[0].shape[0]
                # index by element in history
                for i in xrange(1, len(path_points)):
                    # here we need to draw k arrows
                    #pathpoints[i] is a (k,2) nd-array
                    # iterate over k agents
                    for j in xrange(k):
                        prev = path_points[i - 1]
                        current = path_points[i]
                        axes.arrow(prev[j,0], prev[j,1],
                                       current[j,0] - prev[j,0],
                                       current[j,1] - prev[j,1], edgecolor='red')

        if not save_path == None:
            plt.savefig(save_path + ".png")
        if display: plt.show()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    # Evaluate on single variable MVnormal
    y_vals = np.arange(-10, 10.1, 0.1)
    x_vals = np.arange(10, 30.1, 0.5)
    X, Y = np.meshgrid(x_vals, y_vals)

    print Y
    print X

    # Define synthetic data
    field = multivariate_normal(mean=[20, -5], cov=[[4, 0], [0, 25]])
    vec_field = np.vectorize(lambda x, y: field.pdf([x, y]))

    # Evaluate across synthetic data
    vals = vec_field(X, Y)
    print vals.shape

    # Display synthetic data
    v2d = Vis2d()
    v2d.MapPlot(grid_extent=[10, 30, -10, 10], ground_truth=vals,
                path_points=[(15, 5), (14, 5), (13, 5), (13, 4), (13, 3)])
