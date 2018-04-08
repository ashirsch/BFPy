from bokeh.layouts import layout, widgetbox
from bokeh.models import RangeSlider, Button, MultiSelect, TextInput, Label, BoxSelectTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
import os
import spe_loader


class BokehLoader(object):

    def __init__(self):
        self.spe_file = None
        self.full_sensor_data = None
        self.selection_data = None

        # setup widgets
        self.directory_input = TextInput(placeholder='Directory', value=os.path.join(os.getcwd(), 'data'))
        self.show_files_button = Button(label='Show Files', button_type='primary')
        self.file_view = MultiSelect(size=5)
        self.open_file_button = Button(label='Open File', button_type='warning')
        self.update_selection_button = Button(label='Update Selection', button_type='success')
        self.selection_range = RangeSlider(start=0, end=1, value=(0,1), step=1, title='Selected Rows')

        # connect button callbacks
        self.show_files_button.on_click(self.update_file_browser)
        self.open_file_button.on_click(self.open_file_callback)
        self.update_selection_button.on_click(self.update_selection)
        self.selection_range.on_change('value', self.selection_range_callback)

        # setup plots
        self.full_sensor_image = figure(x_range=(0, 1), y_range=(0, 1023), tools='pan,box_zoom,wheel_zoom,reset',
                                        plot_width=512, plot_height=512)
        self.full_sensor_image_label = Label(x=0.1, y=0.1, text='Source Data',
                                             text_font_size='36pt', text_color='#eeeeee')
        self.full_sensor_image.add_tools(BoxSelectTool(dimensions='height'))
        self.full_sensor_image.grid.grid_line_color = None
        self.full_sensor_image.xaxis.major_tick_line_color = None
        self.full_sensor_image.xaxis.minor_tick_line_color = None
        self.full_sensor_image.yaxis.major_tick_line_color = None
        self.full_sensor_image.yaxis.minor_tick_line_color = None
        self.full_sensor_image.xaxis.major_label_text_font_size = '0pt'
        self.full_sensor_image.yaxis.major_label_text_font_size = '0pt'
        self.selection_lines_coords = ColumnDataSource(data=dict(x=[[0, 1], [0, 1]], y=[[0, 0], [1, 1]]))

        self.selection_image = figure(x_range=(0,1), y_range=(0, 1), tools='wheel_zoom',plot_width=1024,plot_height=180)
        self.selection_image_label = Label(x=0.1, y=0.2, text='Selection Region', text_font_size='36pt',
                                           text_color='#eeeeee')
        self.selection_image.grid.grid_line_color = None

        # build the layout
        controls = [self.directory_input, self.show_files_button, self.file_view, self.open_file_button, self.selection_range, self.update_selection_button]
        widgets = widgetbox(*controls, width=500)
        self.layout = layout(children=[[widgets],
                                       [self.full_sensor_image, self.selection_image]],
                            sizing_mode='fixed')

        # set defaults
        self.initialize_ui()

    def update_file_browser(self):
        active_dir = self.directory_input.value
        if os.path.isdir(active_dir):
            self.file_view.options = [f for f in os.listdir(active_dir) if os.path.isfile(os.path.join(active_dir, f))]
        else:
            self.file_view.options = ['INVALID DIRECTORY']

    def update_full_sensor_image(self):
        self.full_sensor_image.y_range.start = 0
        self.full_sensor_image.y_range.end = self.full_sensor_data.shape[0] - 1
        self.full_sensor_image.image([self.full_sensor_data], 0, 0,
                                     1, self.full_sensor_data.shape[0], palette='Inferno256')
        self.selection_range.start = 0
        self.selection_range.end = self.full_sensor_data.shape[0] - 1
        self.selection_range.value = (0, self.full_sensor_data.shape[0] - 1)
        self.full_sensor_image.multi_line(xs='x',
                                          ys='y', source=self.selection_lines_coords,
                                          color='red', line_width=5, alpha=0.6)

    def update_selection(self):
        self.selection_image_label.text = ''
        self.selection_image.title.text = 'Selected Data'
        ylim = (int(self.selection_range.value[0]), int(self.selection_range.value[1]))
        self.selection_data = self.full_sensor_data[ylim[0]:ylim[1]+1,:]
        self.selection_image.y_range.start = 0
        self.selection_image.y_range.end = ylim[1] - ylim[0]
        self.selection_image.image([self.selection_data], 0, 0,
                                   1, self.selection_image.y_range.end, palette='Inferno256')

    def open_file_callback(self):
        path_to_file = os.path.join(self.directory_input.value, self.file_view.value[0])
        if not os.path.isfile(path_to_file):
            return
        self.spe_file = spe_loader.load_from_files([path_to_file])
        self.full_sensor_data = self.spe_file.data[0][0]
        self.full_sensor_image_label.text = ''
        self.full_sensor_image.title.text = 'Source Data'
        self.update_full_sensor_image()

    def selection_range_callback(self, attr, old, new):
        ylim = (int(self.selection_range.value[0]), int(self.selection_range.value[1]))
        x = [[0,1],[0,1]]
        y = [[ylim[0], ylim[0]],
             [ylim[1], ylim[1]]]
        self.selection_lines_coords.data = dict(x=x, y=y)

    def initialize_ui(self):
        self.full_sensor_image.add_layout(self.full_sensor_image_label)
        self.selection_image.add_layout(self.selection_image_label)
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
