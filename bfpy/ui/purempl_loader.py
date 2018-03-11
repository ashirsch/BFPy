import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
from PyQt5.QtWidgets import QFileDialog
import spe_loader


class LoaderUI(object):

    def __init__(self):
        self.full_sensor_data = None

        freqs = np.arange(2, 20, 3)

        fig, self.main_ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        # t = np.arange(0.0, 1.0, 0.001)
        # s = np.sin(2 * np.pi * freqs[0] * t)
        # l, = plt.plot(t, s, lw=2)

        # Make open, load, draw buttons
        axopen = plt.axes([0.05, 0.05, 0.1, 0.075])
        axload = plt.axes([0.81, 0.05, 0.1, 0.075])
        axdraw = plt.axes([.5, .05, .1, .075])
        bload = Button(axload, 'Load')
        bload.on_clicked(self._load_callback)
        bopen = Button(axopen, 'Open')
        bopen.on_clicked(self._open_callback)
        bdraw = Button(axdraw, 'Full Lambda')
        bdraw.on_clicked(self._draw_callback)

        print("\n      click  -->  release")

        self.RS = RectangleSelector(self.main_ax, self._line_select_callback,
                                                drawtype='box', useblit=False,
                                                button=[1, 3],  # don't use middle button
                                                minspanx=1, minspany=0.1,
                                                spancoords='data',
                                                interactive=True)
        plt.connect('key_press_event', self._toggle_selector)
        plt.show()

    def _line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))


    def _toggle_selector(self, event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.RS.active:
            print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)

    def _load_callback(self, event):
        print('Load clicked')


    def _open_callback(self, event):
        # calls general observation load function
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        # file_dialog.setFilter("SPE files (*.spe)")
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            spe_data = spe_loader.load_from_files(filenames)
            self.full_sensor_data = spe_data.data[0][0]
            self._image_spe_data()
        else:
            return
        print('Open clicked')

    def _draw_callback(self, event):
        self.RS.to_draw.set_visible(True)
        self.RS.extents = (0, 1024, self.RS.extents[2], self.RS.extents[3])

    def _image_spe_data(self):
        self.main_ax.clear()
        self.RS = RectangleSelector(self.main_ax, self._line_select_callback,
                                    drawtype='box', useblit=False,
                                    button=[1, 3],  # don't use middle button
                                    minspanx=1, minspany=0.1,
                                    spancoords='data',
                                    interactive=True)
        self.main_ax.imshow(self.full_sensor_data)


if __name__ == "__main__":
    LoaderUI()
