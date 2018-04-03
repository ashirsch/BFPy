from bokeh.layouts import layout, widgetbox
from bokeh.models import RangeSlider, Button, MultiSelect, TextInput
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.server.server import Server
import os
import spe_loader


class BokehLoader(object):

    def __init__(self):
        # setup widgets
        self.directory_input = TextInput(placeholder='Directory', value=os.path.join(os.getcwd(), 'data'))
        self.show_files_button = Button(label='Show Files', button_type='primary')
        self.file_view = MultiSelect(size=5)
        self.open_file_button = Button(label='Open File', button_type='warning')

        # connect button callbacks
        self.show_files_button.on_click(self.update_file_browser)
        self.open_file_button.on_click(self.open_file_callback)

        # build the layout
        controls = [self.directory_input, self.show_files_button, self.file_view, self.open_file_button]
        widgets = widgetbox(*controls, sizing_mode='fixed')
        self.layout = layout(children=[widgets],
                   sizing_mode='fixed')

        # set defaults
        self.initialize_ui()

    def update_file_browser(self):
        active_dir = self.directory_input.value
        if os.path.isdir(active_dir):
            self.file_view.options = [f for f in os.listdir(active_dir) if os.path.isfile(os.path.join(active_dir, f))]
        else:
            self.file_view.options = ['INVALID DIRECTORY']


    def open_file_callback(self):
        path_to_file = os.path.join(self.directory_input.value, self.file_view.value[0])
        if not os.path.isfile(path_to_file):
            return
        spe_file = spe_loader.load_from_files([path_to_file])
        full_sensor_data = spe_file.data[0][0]


    def initialize_ui(self):
        self.update_file_browser()


def modify_doc(doc):
    doc.add_root(loader.layout)

loader = BokehLoader()
server = Server({'/': modify_doc}, num_procs=1)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
