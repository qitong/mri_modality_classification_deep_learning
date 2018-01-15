import glob
from pydcmio.dcmconverter.converter import generate_config
from pydcmio.dcmconverter.converter import dcm2nii
from nipype.interfaces.dcm2nii import Dcm2niix
import dicom2nifti
import nibabel as nib
import numpy as np
import cv2
import os

def bounding_box(img):
    '''
    Slice out zero values and keep only rows and cols with non-zero values
    :param img:
    :return:
    '''
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]

def resize_and_pad(img, size, padColor = 0):
    '''
    resize image while keeping aspect ratio, pad the shorter side of the image with padColor
    :param img:
    :param size:
    :param padColor:
    :return:
    '''
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # enlarge image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: #horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def pixel_array_to_resized_jpeg(pixel_array, jpeg_file_path, resize_shape=(128, 128)):
    """
    Function to convert an dicom pixel_array file to a jpeg image file

    :param dicom_file_path:
    :param jpeg_file_path:
    :return:
    """
    image_2d = []
    max_val = 0
    # extract the largest pixel value
    #print(pixel_array.shape)
    for row in pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col > max_val:
                max_val = col
        image_2d.append(pixels)

    # if image is completely dark discard this image
    if max_val == 0:
        return

    # rescaling greyscale value between 0-255
    image_2d_normalized = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = int((float(col)/float(max_val))*255.0)
            row_scaled.append(col_scaled)
        image_2d_normalized.append(row_scaled)

    image_2d_normalized = np.array(image_2d_normalized)

    # turn gray scale with 1 channel to rgb with 3 channel
    shape = image_2d_normalized.shape
    img = np.zeros((shape[0], shape[1], 3))
    img[:,:,0] = image_2d_normalized
    img[:,:,1] = image_2d_normalized
    img[:,:,2] = image_2d_normalized

    # resize and pad expects a 3 channel rgb numpy array
    scaled_img = resize_and_pad(img, resize_shape, 0)
    cv2.imwrite(jpeg_file_path, scaled_img)

def convert_dcm_2_nii(dcm_folder):
    for subject_folder in glob.glob(dcm_folder):
        output_path = subject_folder
        config_file = generate_config(
            output_path, anonymized=False, gzip=True, add_date=False,
            add_acquisition_number=False, add_protocol_name=False,
            add_patient_name=False, add_source_filename=True,
            begin_clip=0, end_clip=0)
        (files, reoriented_files, reoriented_and_cropped_files, bvecs, bvals) \
            = dcm2nii(subject_folder, o=output_path, b=config_file)

def convert_dcm_2_nii_x(dcm_folder, output_folder):
    converter = Dcm2niix()
    converter.inputs.source_dir = dcm_folder
    converter.inputs.output_dir = output_folder
    converter.inputs.compress = 'i'
    converter.run()

def convert_dicom_2_nifti(dcm_folder, output_folder):
    dicom2nifti.convert_directory(dcm_folder, output_folder)

def convert_nii_to_jpeg(filename, output_folder, output_name, shape=(128,128)):
    nii = nib.load(filename)
    pixel_array = nii.get_data()
    layer_count = pixel_array.shape[2]
    for layer_num in range(int((1/3)*layer_count), int((4/5)*layer_count)):
        file_path = os.path.join(output_folder, str(output_name) + '.jpeg')
        #print(file_path)
        pixel_array_to_resized_jpeg(pixel_array[:,:,layer_num], file_path, shape)
        output_name += 1
    return output_name

