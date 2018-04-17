import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector, TextBox, CheckButtons, SpanSelector
from PyQt5.QtWidgets import QFileDialog
import spe_loader


class LoaderUI(object):

    def __init__(self):
        self.full_sensor_data = None
        self.selected_data = None
        self.spe_file = None
        self.selector = None
        self.pol_angle = None
        self.success = False

        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'style', 'custom-wa.mplstyle')
        plt.style.use(filename)

        fig = plt.figure()
        fig.set_size_inches(20, 12, forward=True)
        fig.canvas.set_window_title('Load Data')

        grid_shape = (16, 28)
        # Make open, load, draw buttons
        self.full_sensor_ax = plt.subplot2grid(grid_shape, (0,0), colspan=13, rowspan=13)
        self.selected_ax = plt.subplot2grid(grid_shape, (0, 15), colspan=13, rowspan=4)
        axopen = plt.subplot2grid(grid_shape, (14,4), colspan=4)
        axload = plt.subplot2grid(grid_shape, (14,20), colspan=4)
        axfull_lambda = plt.subplot2grid(grid_shape, (11,16), colspan=4)
        axpix_min = plt.subplot2grid(grid_shape, (7,16), colspan=4)
        axpix_max = plt.subplot2grid(grid_shape, (7,23), colspan=4)
        axrefresh = plt.subplot2grid(grid_shape, (11, 23), colspan=4)
        axpol_angle = plt.subplot2grid(grid_shape, (9, 23), colspan=4)

        bload = Button(axload, 'Load Selected', color='0.25', hovercolor='0.3')
        bload.on_clicked(self._load_callback)
        bopen = Button(axopen, 'Open File', color='0.25', hovercolor='0.3')
        bopen.on_clicked(self._open_callback)

        self.chk_full_lambda = CheckButtons(axfull_lambda, ['Full Lambda'], [True])
        self.chk_full_lambda.on_clicked(self._full_lambda_callback)

        self.ypix_min = TextBox(axpix_min, 'Sel. Lower \nbound ', '0', color='0.25', hovercolor='0.3')

        self.ypix_max = TextBox(axpix_max, 'Sel. Upper \nbound ', '0', color='0.25', hovercolor='0.3')

        self.refresh_selection = Button(axrefresh, 'Refresh Selection', color='0.25', hovercolor='0.3')
        self.refresh_selection.on_clicked(self._refresh_selection_callback)

        self.txt_pol_angle = TextBox(axpol_angle, 'Pol. Angle \n (deg) ', '0', color='0.25', hovercolor='0.3')

        self._full_lambda_callback(None)

        plt.show(block=True)

    def _rect_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

    def _span_select_callback(self, ymin, ymax):
        ymin = int(np.floor(ymin))
        ymax = int(np.ceil(ymax))
        print("(%3.2f) --> (%3.2f)" % (ymin, ymax))
        self.selected_data = self.full_sensor_data[ymin:ymax+1,:]
        self.ypix_min.set_val(str(ymin))
        self.ypix_max.set_val(str(ymax))
        self._image_selected_data(ymin, ymax)

    def _load_callback(self, event):
        print('Load clicked')
        if self.spe_file is not None and self.selected_data is not None:
            self.pol_angle = float(self.txt_pol_angle.text)
            self.success = True
            plt.close()
        else:
            print('Invalid loader state: must have loaded file and selected data')

    def _open_callback(self, event):
        # calls general observation load function
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        # file_dialog.setFilter("SPE files (*.spe)")
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            self.spe_file = spe_loader.load_from_files(filenames)
            self.full_sensor_data = self.spe_file.data[0][0]
            self._image_full_sensor_data()
        else:
            return
        print('Open clicked')

    def _full_lambda_callback(self, event):
        if self.selector is not None:
            self.selector.set_visible(False)
        chk_status = self.chk_full_lambda.get_status()
        if chk_status[0]:
            self.selector = SpanSelector(self.full_sensor_ax, self._span_select_callback, direction='vertical',
                                         minspan=5, useblit=True, span_stays=False,
                                         rectprops=dict(facecolor='red', alpha=0.2))
        else:
            self.selector = RectangleSelector(self.full_sensor_ax, self._rect_select_callback,
                                              drawtype='box', useblit=True,
                                              button=[1, 3],  # don't use middle button
                                              minspanx=1, minspany=0.1,
                                              spancoords='data',
                                              interactive=True)

    def _refresh_selection_callback(self, event):
        ymin = int(self.ypix_min.text)
        ymax = int(self.ypix_max.text)
        self._span_select_callback(ymin, ymax)

    def _image_full_sensor_data(self):
        self.full_sensor_ax.clear()
        self._full_lambda_callback(None)
        self.full_sensor_ax.imshow(self.full_sensor_data, aspect='auto')
        self.full_sensor_ax.set_title('Source Data')

    def _image_selected_data(self, ymin, ymax):
        self.selected_ax.clear()
        self.selected_ax.imshow(self.selected_data, aspect='auto')
        title = 'Selected: {0} rows ({1} -> {2})\n' \
                '{3} wavelengths ({4:.3f} -> {5:.3f})'.format(self.selected_data.shape[0], ymin, ymax,
                                                     self.selected_data.shape[1], self.spe_file.wavelength[0],
                                                     self.spe_file.wavelength[-1])
        self.selected_ax.set_title(title)


if __name__ == "__main__":
    LoaderUI()
