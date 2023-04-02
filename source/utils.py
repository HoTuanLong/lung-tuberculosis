from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from libs import *

def dcm2image(dcm_path):
    dcm = dcmread(dcm_path)
    array = apply_voi_lut(dcm.pixel_array, dcm)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        array = np.amax(array) - array

    array = array - np.min(array)
    array = array / np.max(array)
    array = array * 255
    array = array.astype(np.uint8)
    # print(array.shape)

    if len(array.shape) < 3:
        image = np.stack([array] * 3, axis=0)
        image = np.transpose(image, (1, 2, 0))
    else:
        image = array

    return image