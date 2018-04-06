from bokeh.layouts import layout, widgetbox
from bokeh.models import RangeSlider, Button, MultiSelect, TextInput
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.server.server import Server
import os
import spe_loader


def modify_doc(doc):
    def update_file_browser():
        active_dir = directory_input.value
        if os.path.isdir(active_dir):
            file_view.options = [f for f in os.listdir(active_dir) if os.path.isfile(os.path.join(active_dir, f))]
        else:
            file_view.options = ['INVALID DIRECTORY']


    def open_file_callback():
        path_to_file = os.path.join(directory_input.value, file_view.value[0])
        if not os.path.isfile(path_to_file):
            return
        spe_file = spe_loader.load_from_files([path_to_file])
        full_sensor_data = spe_file.data[0][0]


    def initialize():
        update_file_browser()

    # setup widgets
    directory_input = TextInput(placeholder='Directory', value=os.path.join(os.getcwd(), 'data'))
    show_files_button = Button(label='Show Files', button_type='primary')
    file_view = MultiSelect(size=5)
    open_file_button = Button(label='Open File', button_type='warning')

    # connect button callbacks
    show_files_button.on_click(update_file_browser)
    open_file_button.on_click(open_file_callback)

    controls = [directory_input, show_files_button, file_view, open_file_button]

    widgets = widgetbox(*controls, sizing_mode='fixed')

    l = layout(children=[widgets],
               sizing_mode='fixed')

    initialize()
    doc.add_root(l)

server = Server({'/': modify_doc}, num_procs=1)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
