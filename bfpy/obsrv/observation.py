from bfpy.ui import LoaderUI

# noinspection PyPep8Naming
class Observation(object):

    def __init__(self):
        self.filepath = None
        self.data = None
        self.wavelength = None
        self.loaded = False
        self.pol_angle = None

    @property
    def n_frames(self):
        if self.data is not None:
            return self.data.shape[2]
        else:
            return 0

    @property
    def dispersed_pixel_count(self):
        if self.data is not None:
            return self.data.shape[1]
        else:
            return 0

    @property
    def angular_pixel_count(self):
        if self.data is not None:
            return self.data.shape[0]
        else:
            return 0

    # TODO: Allow for loading of multiple frames (and sum and average options)
    def load(self):
        loader = LoaderUI()
        if loader.success:
            self.load_from_array(loader.selected_data, loader.spe_file.wavelength, loader.pol_angle, loader.spe_file.filepath)
        return self.loaded

    def load_from_array(self, numpy_data, wavelength, pol_angle, filepath=None):
        # analyzes numpy_data and puts into proper place
        self.data = numpy_data
        self.wavelength = wavelength
        self.pol_angle = pol_angle
        if filepath is not None:
            self.filepath = filepath
        self.loaded = True
