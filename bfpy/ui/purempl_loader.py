import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector, TextBox, CheckButtons, SpanSelector
from PyQt5.QtWidgets import QFileDialog
import spe_loader


class LoaderUI(object):

    def __init__(self, pol_angle=None):
        self.full_sensor_data = None
        self.selected_data = None
        self.spe_file = None
        self.selector = None
        self.__pol_angle = pol_angle
        self.success = False

        fig, (self.full_sensor_ax, self.selected_ax) = plt.subplots(1, 2)
        plt.subplots_adjust(bottom=0.4)
        fig.canvas.set_window_title('Load {0} Deg. Polarized Data'.format(self.__pol_angle))

        # Make open, load, draw buttons
        axopen = plt.axes([0.05, 0.05, 0.2, 0.07])
        axload = plt.axes([0.75, 0.05, 0.2, 0.07])
        axfull_lambda = plt.axes([0.40, 0.05, 0.2, 0.07])
        bload = Button(axload, 'Load Selected')
        bload.on_clicked(self._load_callback)
        bopen = Button(axopen, 'Open File')
        bopen.on_clicked(self._open_callback)

        self.chk_full_lambda = CheckButtons(axfull_lambda, ['Full Lambda'], [True])
        self.chk_full_lambda.on_clicked(self._full_lambda_callback)

        axpix_min = plt.axes([0.40, 0.17, 0.05, 0.07])
        self.ypix_min = TextBox(axpix_min, 'Sel. Lower \nbound ', '0')

        axpix_max = plt.axes([0.55, 0.17, 0.05, 0.07])
        self.ypix_max = TextBox(axpix_max, 'Sel. Upper \nbound ', '0')

        axrefresh = plt.axes([0.75, 0.17, 0.2, 0.07])
        self.refresh_selection = Button(axrefresh, 'Refresh Selection')
        self.refresh_selection.on_clicked(self._refresh_selection_callback)

        axpol_angle = plt.axes([0.2, 0.17, 0.05, 0.07])
        txt_pol_angle = TextBox(axpol_angle, 'Pol. Angle \n (deg) ', '0')

        print("\n      click  -->  release")

        self._full_lambda_callback(None)
        plt.connect('key_press_event', self._toggle_selector)

        plt.show(block=True)

    def _rect_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

    def _span_select_callback(self, ymin, ymax):
        print("(%3.2f) --> (%3.2f)" % (ymin, ymax))
        ymin = int(np.floor(ymin))
        ymax = int(np.ceil(ymax))
        self.selected_data = self.full_sensor_data[ymin:ymax+1,:]
        self.ypix_min.set_val(str(ymin))
        self.ypix_max.set_val(str(ymax))
        self._image_selected_data(ymin, ymax)

    def _toggle_selector(self, event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.selector.active:
            print(' RectangleSelector deactivated.')
            self.selector.set_active(False)
        if event.key in ['A', 'a'] and not self.selector.active:
            print(' RectangleSelector activated.')
            self.selector.set_active(True)

    def _load_callback(self, event):
        print('Load clicked')
        if self.spe_file is not None and self.selected_data is not None:
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
                                         minspan=5, useblit=False, span_stays=False,
                                         rectprops=dict(facecolor='red', alpha=0.2))
        else:
            self.selector = RectangleSelector(self.full_sensor_ax, self._rect_select_callback,
                                              drawtype='box', useblit=False,
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
        self.full_sensor_ax.imshow(self.full_sensor_data)
        self.full_sensor_ax.set_title('Source Data')

    def _image_selected_data(self, ymin, ymax):
        self.selected_ax.clear()
        self.selected_ax.imshow(self.selected_data)
        title = 'Selected: {0} rows ({1} -> {2})\n' \
                '{3} wavelengths ({4:.3f} -> {5:.3f})'.format(self.selected_data.shape[0], ymin, ymax,
                                                     self.selected_data.shape[1], self.spe_file.wavelength[0],
                                                     self.spe_file.wavelength[-1])
        self.selected_ax.set_title(title)


if __name__ == "__main__":
    LoaderUI()
