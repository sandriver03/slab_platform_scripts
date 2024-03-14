"""Draw polygon regions of interest (ROIs) in matplotlib images,
similar to Matlab's roipoly function.
"""

import sys
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button

logger = logging.getLogger(__name__)

warnings.simplefilter('always', DeprecationWarning)


def deprecation(message):
    warnings.warn(message, DeprecationWarning)


class RoiPoly:

    def __init__(self, data=None, fig=None, ax=None, color='b',
                 roicolor=None, show_fig=True, close_fig=True, **kwargs):
        """
        Parameters
        ----------
        data: 2D np.array
            The image on which the ROI is chosen
        fig: matplotlib figure
            Figure on which to create the ROI
        ax: matplotlib axes
            Axes on which to draw the ROI
        color: str
           Color of the ROI
        roicolor: str
            deprecated, use `color` instead
        show_fig: bool
            Display the figure upon initializing a RoiPoly object
        close_fig: bool
            Close the figure after finishing ROI drawing
        kwargs: currently 'cmap' and 'clim' are used
        """

        if roicolor is not None:
            deprecation("Use 'color' instead of 'roicolor'!")
            color = roicolor

        if data is None:
            # check current figure for data
            if fig is None:
                fig = plt.gcf()
            if ax is None:
                # assuming current figure only contains one axes
                if len(fig.axes) > 1:
                    raise ValueError('more than one axes in current figure! please specify which axes to use')
                elif not fig.axes:
                    raise ValueError('no axes is specified')
                ax = fig.axes[0]
            # ax should contain an image
            if not ax.images:
                raise ValueError('the axes does not contain an image')
            elif len(ax.images) > 1:
                raise ValueError('the axes contains more than 1 images')
            else:
                data = ax.images[0].get_array()
        else:
            if fig is None:
                if ax is None:
                    fig = plt.figure()
                    ax = fig.subplots()
                else:
                    fig = ax.figure
            # plot the figure if not there
            if not ax.images:
                clim = None
                cmap = None
                if 'cmap' in kwargs:
                    cmap = kwargs['cmap']
                if 'clim' in kwargs:
                    clim = kwargs['clim']
                ax.imshow(data, cmap=cmap, clim=clim)

        self.data = data
        self.start_point = []
        self.end_point = []
        self.previous_point = []
        self.x = []
        self.y = []
        self.line = None
        self.completed = False  # Has ROI drawing completed?
        self.color = color
        self.fig = fig
        self.ax = ax
        self.close_figure = close_fig

        # Mouse event callbacks
        self.__cid1 = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.__motion_notify_callback)
        self.__cid2 = self.fig.canvas.mpl_connect(
            'button_press_event', self.__button_press_callback)

        if show_fig:
            self.show_figure()

    @staticmethod
    def show_figure():
        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show(block=True)

    def get_mask(self, current_image=None):
        if not current_image:
            current_image = self.data
        ny, nx = np.shape(current_image)
        poly_verts = ([(self.x[0], self.y[0])]
                      + list(zip(reversed(self.x), reversed(self.y))))
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        roi_path = MplPath(poly_verts)
        grid = roi_path.contains_points(points).reshape((ny, nx))
        return grid

    def get_boundary(self):
        """get the enclosed boundary of the roi as a series of points"""
        xs = self.x.copy()
        ys = self.y.copy()
        xs.append(xs[0])
        ys.append(ys[0])
        return xs, ys

    def display_roi(self, **linekwargs):
        line = plt.Line2D(self.x + [self.x[0]], self.y + [self.y[0]],
                          color=self.color, **linekwargs)
        ax = plt.gca()
        ax.add_line(line)
        plt.draw()

    def get_mean_and_std(self, current_image):
        mask = self.get_mask(current_image)
        mean = np.mean(np.extract(mask, current_image))
        std = np.std(np.extract(mask, current_image))
        return mean, std

    def display_mean(self, current_image, **textkwargs):
        mean, std = self.get_mean_and_std(current_image)
        string = "%.3f +- %.3f" % (mean, std)
        plt.text(self.x[0], self.y[0],
                 string, color=self.color,
                 bbox=dict(facecolor='w', alpha=0.6), **textkwargs)

    def __motion_notify_callback(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if ((event.button is None or event.button == 1) and
                    self.line is not None):
                # Move line around
                x_data = [self.previous_point[0], x]
                y_data = [self.previous_point[1], y]
                logger.debug("draw line x: {} y: {}".format(x_data, y_data))
                self.line.set_data(x_data, y_data)
                self.fig.canvas.draw()

    def __button_press_callback(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if event.button == 1 and event.dblclick is False:
                logger.debug("Received single left mouse button click")
                if self.line is None:  # If there is no line, create a line
                    self.line = plt.Line2D([x, x], [y, y],
                                           marker='o', color=self.color)
                    self.start_point = [x, y]
                    self.previous_point = self.start_point
                    self.x = [x]
                    self.y = [y]

                    ax.add_line(self.line)
                    self.fig.canvas.draw()
                    # Add a segment
                else:
                    # If there is a line, create a segment
                    x_data = [self.previous_point[0], x]
                    y_data = [self.previous_point[1], y]
                    logger.debug(
                        "draw line x: {} y: {}".format(x_data, y_data))
                    self.line = plt.Line2D(x_data, y_data,
                                           marker='o', color=self.color)
                    self.previous_point = [x, y]
                    self.x.append(x)
                    self.y.append(y)

                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()

            elif (((event.button == 1 and event.dblclick is True) or
                   (event.button == 3 and event.dblclick is False)) and
                  self.line is not None):
                # Close the loop and disconnect
                logger.debug("Received single right mouse button click or "
                             "double left click")
                self.fig.canvas.mpl_disconnect(self.__cid1)
                self.fig.canvas.mpl_disconnect(self.__cid2)

                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None
                self.completed = True

                if not sys.flags.interactive and self.close_figure:
                    #  Figure has to be closed so that code can continue
                    plt.close(self.fig)


class MultiRoi:
    def __init__(self, data=None,
                 fig=None, ax=None,
                 roi_names=None,
                 color_cycle=('b', 'g', 'r', 'c', 'm', 'y', 'k')
                 ):
        """
        Parameters
        ----------
        data: 2D np.array
            The image on which the ROI is chosen
        fig: matplotlib figure
            Figure on which to draw the ROIs
        ax: matplotlib axes
           Axes on which to draw the ROIs
        roi_names: list of str
            Optional names for the ROIs to draw.
            The ROIs can later be retrieved by using these names as keys for
            the `self.rois` dictionary. If None, consecutive numbers are used
            as ROI names
        color_cycle: list of str
            List of matplotlib colors for the ROIs
        """

        if data is None:
            # check current figure for data
            if fig is None:
                fig = plt.gcf()
            if ax is None:
                # assuming current figure only contains one axes
                if len(fig.axes) > 1:
                    raise ValueError('more than one axes in current figure! please specify which axes to use')
                ax = fig.axes[0]
            # ax should contain an image
            if not ax.images:
                raise ValueError('the axes does not contain an image')
            elif len(ax.images) > 1:
                raise ValueError('the axes contains more than 1 images')
            else:
                data = ax.images[0].get_array()
        else:
            if fig is None:
                if ax is None:
                    fig = plt.figure()
                    ax = fig.subplots()
                else:
                    fig = ax.figure
            # plot the figure if not there
            if not ax.images:
                ax.imshow(data)

        self.data = data
        self.color_cycle = color_cycle
        self.roi_names = roi_names
        self.fig = fig
        self.ax = ax
        self.rois = {}

        self.make_buttons()

    def make_buttons(self):
        ax_add_btn = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_finish_btn = plt.axes([0.81, 0.02, 0.1, 0.04])
        btn_finish = Button(ax_finish_btn, 'Finish')
        btn_finish.on_clicked(self.finish)
        btn_add = Button(ax_add_btn, 'New ROI')
        btn_add.on_clicked(self.add)
        plt.show(block=True)

    def add(self, event):
        """"Add a new ROI"""

        # Only draw a new ROI if the previous one is completed
        if self.rois:
            if not all(r.completed for r in self.rois.values()):
                return

        count = len(self.rois)
        idx = count % len(self.color_cycle)
        logger.debug("Creating new ROI {}".format(count))
        if self.roi_names is not None and idx < len(self.roi_names):
            roi_name = self.roi_names[idx]
        else:
            roi_name = str(count + 1)

        self.ax.set_title("Draw ROI '{}'".format(roi_name))
        plt.draw()
        roi = RoiPoly(color=self.color_cycle[idx],
                      fig=self.fig,
                      ax=self.ax,
                      data=self.data,
                      close_fig=False,
                      show_fig=False)
        self.rois[roi_name] = roi

    def finish(self, event):
        logger.debug("Stop ROI drawing")
        plt.close(self.fig)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                               '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                        level=logging.INFO)

    # Create image
    img = np.ones((100, 100)) * range(0, 100)

    # Show the image
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    plt.title("left click: line segment         right click or double click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)

    # Show the image with the first ROI
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    roi1.display_roi()
    plt.title('draw second ROI')
    plt.show(block=False)
