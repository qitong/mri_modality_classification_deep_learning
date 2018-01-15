import nibabel as nib
import glob
import os
import mri_utilities
import shutil
import keras_bottleneck_multiclass
from collections import defaultdict
import random

# the training files structure section
# this section can be different than the real deployment folder structure
# training
def convert_data_folder_to_nii(data_folder):
    for patient_folder in glob.glob(os.path.join(data_folder, '*')):
        print(patient_folder)
        mri_utilities.convert_dcm_2_nii_x(patient_folder, patient_folder)
        # there could be a second layer folder or no second layer folder

# generate jpeg files for training
def convert_nii_to_jpeg(data_folder, output_folder):
    for modality_folder in glob.glob(os.path.join(data_folder, '*')):
        modality_folder_name = os.path.basename(modality_folder)
        modality_jpeg_folder = os.path.join(output_folder, modality_folder_name)
        print(modality_jpeg_folder)
        if os.path.exists(modality_jpeg_folder):
            shutil.rmtree(modality_jpeg_folder)
        os.makedirs(modality_jpeg_folder)

        count = 0
        for patient_folder in glob.glob(os.path.join(modality_folder, '*')):
            for ima_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(ima_folder, '*.nii.gz')):
                    print(nii_file)
                    count = mri_utilities.convert_nii_to_jpeg(
                        nii_file, modality_jpeg_folder, count
                        , shape=(keras_bottleneck_multiclass.img_width, keras_bottleneck_multiclass.img_height))

def convert_201801_IDH_nii_to_jpeg():
    #all_data_dir = '/media/mingrui/DATA/datasets/201801-IDH-test'
    #jpeg_data_dir = '/media/mingrui/DATA/datasets/201801-IDH-test-jpeg'
    all_data_dir = '/media/mingrui/DATA/datasets/201801-IDH'
    jpeg_data_dir = '/media/mingrui/DATA/datasets/201801-IDH-jpeg'
    modalities = ['CE', 'T1', 'T2', 'DWI', 'ADC']
    file_types = ['*.nii', '*.nii.gz']

    # save all file paths of a modality inside a dictionary
    modality_dict = defaultdict(list)

    for patient_dir in glob.glob(os.path.join(all_data_dir, '*/')):
        get_modality_file_path(patient_dir, modalities, file_types, modality_dict)

    print(modality_dict)

    # create folders for modalities
    if os.path.exists(jpeg_data_dir):
        shutil.rmtree(jpeg_data_dir)
    os.makedirs(jpeg_data_dir)

    for modality in modalities:
        modality_dir = os.path.join(jpeg_data_dir, modality)
        if os.path.exists(modality_dir):
            shutil.rmtree(modality_dir)
        os.makedirs(modality_dir)

    for modality in modalities:
        count = 0
        modality_dir = os.path.join(jpeg_data_dir, modality)
        for nii_file in modality_dict[modality]:
            print(nii_file)
            try:
                count = mri_utilities.convert_nii_to_jpeg(
                    nii_file, modality_dir, count
                    , shape=(keras_bottleneck_multiclass.img_width, keras_bottleneck_multiclass.img_height))
            except Exception as e:
                print(str(e))


def get_modality_file_path(dir, modalities, file_types, modality_dict):
    print(dir)
    os.chdir(dir)
    files_grabbed = []
    modality_files = []

    for type in file_types:
        files_grabbed.extend(glob.glob(type))

    # check file name
    for file in files_grabbed:
        for modality in modalities:
            if modality in file:
                modality_dict[modality].append(os.path.join(dir, file))

def create_validation_and_test_set():
    train_data_dir = keras_bottleneck_multiclass.train_data_dir
    validation_dir = keras_bottleneck_multiclass.validation_data_dir
    test_dir = keras_bottleneck_multiclass.test_data_dir
    modalities = ['CE', 'T1', 'T2', 'DWI', 'ADC']

    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.makedirs(validation_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # randomly select validation
    for modality in modalities:
        modality_train_dir = os.path.join(train_data_dir, modality)
        modality_validation_dir = os.path.join(validation_dir, modality)
        modality_test_dir = os.path.join(test_dir, modality)
        if os.path.exists(modality_validation_dir):
            shutil.rmtree(modality_validation_dir)
        os.makedirs(modality_validation_dir)
        if os.path.exists(modality_test_dir):
            shutil.rmtree(modality_test_dir)
        os.makedirs(modality_test_dir)
        jpeg_list = [os.path.basename(x) for x in glob.glob(os.path.join(modality_train_dir, '*'))]
        validation_jpeg = random.sample(jpeg_list, int(0.1 * len(jpeg_list)))
        print(validation_jpeg)
        for jpeg in validation_jpeg:
            shutil.move(os.path.join(modality_train_dir, jpeg), os.path.join(modality_validation_dir, jpeg))

        jpeg_list = [os.path.basename(x) for x in glob.glob(os.path.join(modality_train_dir, '*'))]
        test_jpeg = random.sample(jpeg_list, int(0.1 * len(jpeg_list)))
        print(test_jpeg)
        for jpeg in test_jpeg:
            shutil.move(os.path.join(modality_train_dir, jpeg), os.path.join(modality_test_dir, jpeg))

if __name__ == '__main__':
    print('preprocess')
    #convert_data_folder_to_nii('/media/mingrui/DATA/datasets/201801-IDH-test')
    #convert_nii_to_jpeg('/media/mingrui/DATA/datasets/Modality', '/media/mingrui/DATA/datasets/ModalityJpeg')
    #convert_201801_IDH_nii_to_jpeg()
    #create_validation_and_test_set()