import numpy as np
import matplotlib.pyplot as plt


def basis_func_plot(basis, wavelength_ind, crop=False):
    plot_col = basis.basis_matrix[:,wavelength_ind].todense()
    if crop: # TODO: get this working with MD / EQ basis and shorter wavelength ranges
        wavelength_ind = wavelength_ind % basis.basis_parameters.orig_wavelength_count
        begin_ind = int(wavelength_ind-np.floor(basis.basis_parameters.ux_count/2))*basis.basis_parameters.uy_count
        end_ind = begin_ind + basis.basis_parameters.uy_count*basis.basis_parameters.ux_count
        plot_col = plot_col[begin_ind:end_ind]
        plot_width = basis.basis_parameters.ux_count
    elif basis.basis_parameters.orig_wavelength_count > basis.basis_parameters.ux_count:
        plot_width = basis.basis_parameters.orig_wavelength_count
    else:
        plot_width = basis.basis_parameters.ux_count
    plot_col = np.reshape(plot_col, (basis.basis_parameters.uy_count, plot_width), order='F')
    plt.imshow(plot_col)
    plt.show()
