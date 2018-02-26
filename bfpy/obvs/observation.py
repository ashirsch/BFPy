# noinspection PyPep8Naming
class Observation(object):

    def __init__(self):
        self.filepath = ""
        self.data = None
        self.wavelength_range = (0, 0)
        self.NA = 0  # do we need this?

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

    def load(self, filepath=None):
        # makes calls to obvs module, gets back numpy, and puts everything into place
        pass

    def load_from_array(self, numpy_data, NA, wavelength_range):
        # analyzes numpy_data and puts into proper place
        pass

    def set_wavelength_range(self, wavelength_range):
        if (wavelength_range[0] > 0) and (wavelength_range[1] > 0):
            self.wavelength_range = wavelength_range
        else:
            raise ValueError("Invalid wavelength range: {0}".format(wavelength_range))

    def set_NA(self, NA):
        if NA > 0:
            self.NA = NA
        else:
            raise ValueError("Invalid NA: {0}".format(NA))
