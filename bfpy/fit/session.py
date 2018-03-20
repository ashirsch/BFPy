import sys


class Session(object):

    def __init__(self, pol_angles, observations, bases):

        self.__pol_children = None          # list of PolDataSets
        self.model = None
        self.model_parameters = None
        self.vis = None

        try:
            if not isinstance(pol_angles, list):
                assert not isinstance(observations, list) or len(observations) == 1
                assert not isinstance(bases, list) or len(bases) == 1
                pol_angles = [pol_angles]
                observations = [observations]
                bases = [bases]
            else:
                assert len(pol_angles) == len(observations) == len(bases)
        except AssertionError:
            print('Number of polarization angles, observations, bases do not match.')
            raise

        # load data into PolDataSets
        for i in range(len(pol_angles)):
            try:
                assert observations[i].pol_angle == pol_angles[i]
                assert bases[i].pol_angle == pol_angles[i]
            except AssertionError:
                yes = ['y', 'yes']
                no = ['n', 'no', '']
                sys.stdout.write('WARNING: Ordering of polarization angles do not match among arguments.\n'
                                 '    Would you like to load this data into the session anyway? [y/N] ')
                choice = input().lower()
                if choice in yes:
                    continue
                elif choice in no:
                    return
                else:
                    print('Invalid answer. Exiting without making session...\n')
                    return

        self.__pol_children = []
        for i in range(len(pol_angles)):
            self.__pol_children.append(PolDataSet(pol_angles[i], observations[i], bases[i]))

    @property
    def polarization_angles(self):
        angles = []
        for child in self.__pol_children:
            angles.append(child.pol_angle)
        return angles

    @property
    def n_polarizations(self):
        return len(self.__pol_children)

    def data_set(self, pol_angle):
        for child in self.__pol_children:
            if child.pol_angle == pol_angle:
                return child
        raise ValueError('No child with polarization angle {0}'.format(pol_angle))

    def remove_data_set(self, pol_angle):
        child = self.data_set(pol_angle)
        self.__pol_children.remove(child)

    def is_empty(self):
        return not self.__pol_children

    def is_defined(self):
        pass

    def reset(self):
        yes = ['y', 'yes']
        no = ['n', 'no', '']
        sys.stdout.write('WARNING: This action will clear all loaded data, calculated bases, and models.\n'
                         '    Would you still like to proceed? [y/N] ')
        choice = input().lower()
        if choice in yes:
            self.__pol_children.clear()
            self.model = None
            self.vis = None
        elif choice in no:
            return
        else:
            print('Invalid answer. Exiting without resetting...\n')
            return

    def define_fit(self, model, **kwargs):
        # sets model
        pass

    def run(self, open_slit, pad_lambda, fit_all_frames, debug=True):
        pass

    def build_bases(self):
        for angle in self.polarization_angles:
            active_basis = self.data_set(angle).basis
            if active_basis.is_defined and not active_basis.built:
                active_basis.build()

    def visualize(self):
        pass


class PolDataSet(object):

    def __init__(self, pol_angle, obs, basis):
        self.pol_angle = pol_angle
        self.observation = obs
        self.basis = basis
