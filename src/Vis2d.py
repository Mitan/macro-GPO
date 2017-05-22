import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

"""
Heatmap and path plotter
"""


class Vis2d:
    def __init__(self):
        pass

    """
    def MapPlot(self, grid_extent, ground_truth=None, posterior_mean_before=None, posterior_mean_after=None,
                posterior_variance_before=None, posterior_variance_after=None, path_points=None, display=True,
                save_path=None):

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

        grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                        grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left
        mmax = -10 ** 10
        mmin = 10 ** 10
        for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
            if not q is None:
                mmax = max(np.amax(np.amax(q)), mmax)
                mmin = min(np.amin(np.amin(q)), mmin)

        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
        if not ground_truth is None:
            im = axes.flat[0].imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent2,
                                     cmap='Greys', vmin=mmin, vmax=mmax)
            if not path_points is None and path_points:
                for i in xrange(1, len(path_points)):
                    axes.flat[0].arrow(path_points[i - 1][0], path_points[i - 1][1],
                                       path_points[i][0] - path_points[i - 1][0],
                                       path_points[i][1] - path_points[i - 1][1], edgecolor='red')

        if not posterior_mean_before is None:
            im = axes.flat[2].imshow(posterior_mean_before, interpolation='nearest', aspect='auto', extent=grid_extent2,
                                     cmap='Greys', vmin=mmin, vmax=mmax)

        if not posterior_mean_after is None:
            im = axes.flat[3].imshow(posterior_mean_after, interpolation='nearest', aspect='auto', extent=grid_extent2,
                                     cmap='Greys', vmin=mmin, vmax=mmax)

        im2 = None
        if not posterior_variance_before is None:
            im2 = axes.flat[4].imshow(posterior_variance_before, interpolation='nearest', aspect='auto',
                                      extent=grid_extent2)

        if not posterior_variance_after is None:
            im2 = axes.flat[5].imshow(posterior_variance_before, interpolation='nearest', aspect='auto',
                                      extent=grid_extent2)

        cax, kw = mpl.colorbar.make_axes([axes.flat[i] for i in xrange(4)])
        plt.colorbar(im, cax=cax, **kw)

        if not im2 is None:
            cax2, kw2 = mpl.colorbar.make_axes([axes.flat[4], axes.flat[5]])
            plt.colorbar(im2, cax=cax2, **kw2)

        if not save_path is None:
            plt.savefig(save_path + ".png")
        if display: plt.show()
        plt.clf()
        plt.close()
    """

    def MapPlot(self, grid_extent, ground_truth=None, posterior_mean_before=None, posterior_mean_after=None,
                path_points=None, display=True,
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

        # todo remove
        """
        grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                        grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left
        """
        mmax = -10 ** 10
        mmin = 10 ** 10
        for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
            # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
            if q is not  None:
                mmax = max(np.amax(np.amax(q)), mmax)
                mmin = min(np.amin(np.amin(q)), mmin)
        axes = plt.axes()
        # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        if ground_truth is not None:
            im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto',
                             cmap='Greys', vmin=mmin, vmax=mmax)
            if not path_points == None and path_points:
                # batch size
                # path points is a list
                number_of_points = len(path_points)


                for i in xrange(1, number_of_points):
                    # both are batches of points
                    prev = path_points[i-1]
                    current = path_points[i]

                    prev_end = prev[-1, :]
                    current_start = current[0, :]
                    axes.arrow(prev_end[0], prev_end[1],
                               current_start[0] - prev_end[0],
                               current_start[1] - prev_end[1], edgecolor='green')


                    # here we need to draw k - 1 arrows
                    # coz in total there will be k and the first on is already drawn

                    # k should always be equal to batch_size though
                    k = current.shape[0]

                    for j in xrange(0, k-1):
                        # both a locations [x,y]
                        current_point = current[j, :]
                        next_point = current[j+1, :]
                        axes.arrow(current_point[0], current_point[1],
                                   next_point[0] - current_point[0],
                                   next_point[1] - current_point[1], edgecolor='red')


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
