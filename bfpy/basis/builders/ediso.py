import numpy as np
from scipy.sparse import csc_matrix, hstack
from numba import vectorize
from ..fields import field


class EDIsoBuilder:
    """
    :type basis_parameters: BasisParameters
    """

    def __init__(self, basis_parameters):
        self.__bp = basis_parameters
        self.field_set = field.Field(basis_parameters)

    def build(self):
        self.field_set.calculate_fields(["ED", "MD"])
        ed_isometric = isometric_emission(self.__bp.pol_angle,
                                          self.field_set.xpol.ED.x, self.field_set.ypol.ED.x,
                                          self.field_set.xpol.ED.y, self.field_set.ypol.ED.y,
                                          self.field_set.xpol.ED.z, self.field_set.ypol.ED.z)
        md_isometric = isometric_emission(self.__bp.pol_angle,
                                          self.field_set.xpol.MD.x, self.field_set.ypol.MD.x,
                                          self.field_set.xpol.MD.y, self.field_set.ypol.MD.y,
                                          self.field_set.xpol.MD.z, self.field_set.ypol.MD.z)
        # for each ed / md
            # reshape so that each wavelength patter is vectorized (ux_count*uy_count, wavelength_count) (each column a wavelength)
            # load into sparse matrix with spatial offset
        ed_isometric = self.sparse_column_major_offset(ed_isometric)
        md_isometric = self.sparse_column_major_offset(md_isometric)
        # concatenate ED and MD sparse matrices
        basis = hstack([ed_isometric, md_isometric])
        # trim basis based on lambda range TODO: work on x-y flipping
        if self.__bp.trim_w:
            basis = self.basis_trim(basis)
        # return the single sparse matrix
        return basis


    def sparse_column_major_offset(self, matrix):
        # flatten data matrix by column
        flat_matrix = np.reshape(matrix, (self.__bp.ux_count*self.__bp.uy_count*self.__bp.wavelength_count, ), order='F')
        # flat_matrix.flatten('F')
        # generate sparse rows and columns
        rows = sparse_rows(self.__bp.ux_count, self.__bp.uy_count, self.__bp.wavelength_count)
        cols = sparse_cols(self.__bp.ux_count, self.__bp.uy_count, self.__bp.wavelength_count)
        # load into sparse matrix
        return csc_matrix((flat_matrix, (rows, cols)))

    def basis_trim(self, matrix):
        begin_ind = int(self.__bp.uy_count * np.floor(self.__bp.ux_count/2))
        if self.__bp.pad_w:
            begin_ind += int(self.__bp.uy_count * np.floor((self.__bp.ux_count - 1)/2))

        final_pix_row_ind = matrix.shape[0] - 1
        end_ind = int(final_pix_row_ind - self.__bp.uy_count * np.floor((self.__bp.ux_count - 1)/2))
        if self.__bp.pad_w:
            end_ind -= int(self.__bp.uy_count * np.floor(self.__bp.ux_count/2))

        return matrix[begin_ind:(end_ind+1), :]


def sparse_rows(ux_count, uy_count, wavelength_count):
    single_wavelength_rows = np.arange(0, ux_count*uy_count)
    offset = np.arange(0,uy_count*wavelength_count,uy_count)
    rows = single_wavelength_rows.reshape(ux_count*uy_count, 1) + offset
    return rows.flatten('F').astype(int)

def sparse_cols(ux_count, uy_count, wavelength_count):
    single_wavelength_cols = np.arange(0, wavelength_count)
    return np.repeat(single_wavelength_cols, ux_count*uy_count).astype(int)


@vectorize("float64(float64,complex128,complex128,complex128,complex128,complex128,complex128)",
           target='parallel', nopython=True)
def isometric_emission(pol_angle, xDx, yDx, xDy, yDy, xDz, yDz):
    isometric = np.square(np.abs(np.cos(pol_angle) * yDx +
                                 np.sin(pol_angle) * xDx)) + \
                np.square(np.abs(np.cos(pol_angle) * yDy +
                                 np.sin(pol_angle) * xDy)) + \
                np.square(np.abs(np.cos(pol_angle) * yDz +
                                 np.sin(pol_angle) * xDz))
    return isometric