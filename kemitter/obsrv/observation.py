from ..ui import LoaderUI


class Observation(object):
    """Class for storing energy-momentum spectroscopy measurement data.

    Stores all necessary measurement information for fitting and processing.
    Can be used to store any image and wavelength mapping that can be expressed as
    a set of NumPy arrays.

    Attributes:
        data (ndarray): 2D array containing image data with `float` type.
        wavelength (ndarray): 1D array containing wavelength mapping data with `float` type.
        pol_angle (int or float): polarizer angle in degrees.
        filepath (str or None): path to source file containing original data [optional].
        loaded (bool): whether or not data has been loaded by a loading method. Initially set to False.
    """
    def __init__(self):
        self.filepath = None
        self.data = None
        self.wavelength = None
        self.loaded = False
        self.pol_angle = None

    @property
    def n_frames(self):
        """int: the number of frames comprising the observation

        Raises:
            AttributeError: if no data has been loaded

        Warnings:
            Multiple frames in one observation is currently unsupported.
        """
        if self.data is not None:
            return self.data.shape[2]
        else:
            raise AttributeError('No image data has been set.')

    @property
    def dispersed_pixel_count(self):
        """int: the image size in the wavelength (x) dimension

        Raises:
            AttributeError: if no data has been loaded
        """
        if self.data is not None:
            return self.data.shape[1]
        else:
            raise AttributeError('No image data has been set.')

    @property
    def momentum_pixel_count(self):
        """int: the image size in the momentum (y) dimension

        Raises:
             AttributeError: if no data has been loaded
        """
        if self.data is not None:
            return self.data.shape[0]
        else:
            raise AttributeError('No image data has been set.')

    # TODO: Allow for loading of multiple frames (and sum and average options)
    def load(self):
        """
        Launches an interactive loading session.

        Returns (bool):
            True if observation was successfully loaded, False otherwise.

        Notes:
            Opening an interactive loader is a blocking operation, i.e. code will
            stop running until the loader is closed.
        """
        loader = LoaderUI()
        if loader.success:
            self.load_from_array(loader.selected_data, loader.spe_file.wavelength,
                                 loader.pol_angle, loader.spe_file.filepath)
        return self.loaded

    def load_from_array(self, data, wavelength, pol_angle, filepath=None):
        """
        Loads all required observation data into required fields from a generic NumPy array.
        Sets the `loaded` attribute to `True` upon success.

        Args:
            data (ndarray): 2D array containing image data with `float` type
            wavelength (ndarray): 1D array containing wavelength mapping data with `float` type
            pol_angle (int or float): polarizer angle in degrees
            filepath (str or None): path to source file containing original data [optional]
        """
        self.data = data
        self.wavelength = wavelength
        self.pol_angle = pol_angle
        if filepath is not None:
            self.filepath = filepath
        self.loaded = True
