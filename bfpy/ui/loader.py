import sys
from PyQt5.QtWidgets import QApplication, QSpinBox, QPushButton, QGridLayout, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

class LoaderUI(QDialog):

    def __init__(self):
        super().__init__()
        self.title = 'BFPy Observation Loader'
        self.top = 10
        self.left =  10
        self.width = 640
        self.height = 480

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.obs_figure, self.obs_ax = plt.subplots()
        self.canvas = FigureCanvas(self.obs_figure)

        self.choose_button = QPushButton('Choose Data')
        self.choose_button.setObjectName('chooseButton')
        self.choose_button.clicked.connect(self.plot)

        self.select_height = QSpinBox()
        self.select_height.setObjectName('selectHeight')
        self.select_width = QSpinBox()
        self.select_width.setObjectName('selectWidth')
        self.select_top = QSpinBox()
        self.select_top.setObjectName('selectTop')
        self.select_left = QSpinBox()
        self.select_left.setObjectName('selectLeft')

        self.setLayout(self.make_layout())

    def make_layout(self):
        layout = QGridLayout()

        layout.addWidget(self.canvas, 0, 0, 1, 4)
        layout.addWidget(self.choose_button, 1, 0)
        layout.addWidget(self.select_height, 2, 0)
        layout.addWidget(self.select_width, 2, 1)
        layout.addWidget(self.select_top, 2, 2)
        layout.addWidget(self.select_left, 2, 3)

        return layout

    def plot(self):
        ''' plot some random stuff '''
        N = 100000  # If N is large one can see
        x = np.linspace(0.0, 10.0, N)  # improvement by use blitting!

        self.obs_ax.plot(x, +np.sin(.2 * np.pi * x), lw=3.5, c='b', alpha=.7)  # plot something
        # plt.plot(x, +np.cos(.2 * np.pi * x), lw=3.5, c='r', alpha=.5)
        # plt.plot(x, -np.sin(.2 * np.pi * x), lw=3.5, c='g', alpha=.3)

        print("\n      click  -->  release")

        # drawtype is 'box' or 'line' or 'none'


        RS = RectangleSelector(self.obs_ax, self.line_select_callback,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        self.obs_ax.add_patch(RS)

        # refresh canvas
        self.canvas.draw()

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))




if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = LoaderUI()
    main.show()

    sys.exit(app.exec_())






