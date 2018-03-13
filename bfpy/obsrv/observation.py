from bfpy.ui import LoaderUI

# noinspection PyPep8Naming
class Observation(object):

    def __init__(self, pol_angle):
        self.filepath = None
        self.data = None
        self.wavelength = None
        self.__pol_angle = pol_angle
        # self.NA = 0  # do we need this?

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
    def _load(self):
        loader = LoaderUI()
        if loader.success:
            self._load_from_array(loader.selected_data, loader.spe_file.wavelength, loader.spe_file.filepath)
            return True
        else:
            return False

    def _load_from_array(self, numpy_data, wavelength, filepath=None):
        # analyzes numpy_data and puts into proper place
        self.data = numpy_data
        self.wavelength = wavelength
        if filepath is not None:
            self.filepath = filepath

if __name__ == "__main__":
    obs = Observation(0)
    success = obs._load()
    if success:
        print(obs.filepath)
    else:
        print("load failed")
