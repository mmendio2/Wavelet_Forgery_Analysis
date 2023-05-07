import numpy as np
import pywt
import pywt.data
from scipy.stats import skew, kurtosis, f_oneway
from PIL import Image

def split(array, nrows, ncols):
    '''
    Takes an array and splits it into sub-arrays
    :param array: Array/Matrix that you wish to split
    :param nrows: Desired number of rows
    :param ncols: Desired number of columns
    :return: A list of subarrays of size nrows, ncolumns
    '''
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))


def estimator(s_matrix, p_matrix, o_m_1, o_m_2, parent_o_m_1, parent_o_m_2):
    """
    Constructs a linear predictor given matrices and returns the weights that satisfy
    s_matrix = Qw along with the log error.
    :param s_matrix:
    :param p_matrix: parent of s_matrix
    :param o_m_1: matrix of first remaining sub-band
    :param o_m_2: matrix of second remaining sub-band
    :param parent_o_m_1: parent of o_m_1
    :param parent_o_m_2: parent of o_m_2
    :return: Weights and Log Error of estimator
    """
    s_col = np.abs(s_matrix.transpose().flatten())
    s_len = len(s_col)
    s_m_row = len(s_matrix)
    Q = np.array([0.0] * (s_len * 9)).reshape(s_len, 9)
    for i in range(s_len):
        r = i % s_m_row

        Q[i][0] = s_col[i - 1] if r != 0 else 0.0
        Q[i][1] = s_col[i - s_m_row] if i >= s_m_row else 0.0
        Q[i][2] = s_col[i + 1] if (r != (s_m_row - 1)) else 0.0
        Q[i][3] = s_col[i + s_m_row] if i < (s_m_row * (s_m_row - 1)) else 0.0

        c = int(np.floor(i / s_m_row))
        Q[i][4] = o_m_1[r][c]
        Q[i][5] = o_m_2[r][c]
        Q[i][6] = p_matrix[int(np.floor(r / 2))][int(np.floor(c / 2))]
        Q[i][7] = parent_o_m_1[int(np.floor(r / 2))][int(np.floor(c / 2))]
        Q[i][8] = parent_o_m_2[int(np.floor(r / 2))][int(np.floor(c / 2))]
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Q), Q)), np.transpose(Q)), s_col)
    Qw = np.matmul(Q, w)

    Qw[np.where(Qw == 0.0)] = np.float32(1e-8)
    s_col[np.where(s_col == 0.0)] = np.float32(1e-8)
    log_error = np.log2(s_col) - np.log2(np.abs(Qw))
    return w, log_error


def I_matrix(image):
    """
    Takes an image, splits it into 4 subimages, then collects the w and log error
    values for each
    :param image: image to characterize
    :return: complete values of image
    """
    sub_l = int(len(image)/2)
    subimages = split(image, sub_l, sub_l)
    I = np.array([0.0] * (24)).reshape(6, 4)

    for level, subimage in enumerate(subimages):
        LL, (H2, V2, D2), (H1, V1, D1) = pywt.wavedec2(subimage, "db2", mode='per', level=2)
        try:
            w_V1, E_V1 = estimator(V1, V2, D1, H1, D2, H2)
        except np.linalg.LinAlgError:
            pass
        else:
            w_H1, E_H1 = estimator(H1, H2, V1, D1, V2, D2)
            w_D1, E_D1 = estimator(D1, D2, V1, H1, V2, H2)
            I[0][level] = skew(w_V1)
            I[1][level] = skew(w_H1)
            I[2][level] = skew(w_D1)
            I[3][level] = skew(E_V1)
            I[4][level] = skew(E_H1)
            I[5][level] = skew(E_D1)
    return I

def calculate_authenticity(known, test):
    """
    Takes two images, the known signature and the test signature
    and returns if they are statistically the same
    :param known: image of known signature
    :param test: image of signature to test
    :return: True if ANOVA produces >0.95 for any of the tests, false if not
    """
    _, pvalue = f_oneway(known, test, axis = 1)
    return True if (pvalue >= 0.95).any() else False
