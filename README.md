# mri_modality_classification_deep_learning

The problem this tool solves:
Doctors need to organize large amounts of patient scans, convert them from dicom to nifti, and organize them by modality type. Currently, this is all done by hand. Doctors need to take a look at each mri image and decide on the modality. Between some modalities such as T1 and CE, it can be difficult to tell. And there are many many modalites to organize.To make matters worse, sometimes the dicom data are not organized well, this will make organizing mri data a very time consuming test.

The solution to this problem:
Using Keras Tensorflow to classify mri modalities. Currently classifies T1, T2, CE(T1C), DWI, ADC
Automate!

My folder structure:

--- jpeg_train

  |--- T1
  
  |--- T2
  
  |--- CE
  
  |--- ADC
  
  |--- DWI
  
--- jpeg_validation

  |--- T1
  
  |--- T2
  
  |--- CE
  
  |--- ADC
  
  |--- DWI
  
--- jpeg_test

  |--- T1
  
  |--- T2
  
  |--- CE
  
  |--- ADC
  
  |--- DWI
  
I have about 5000 to 6000 training data each. 500 to 600 test and validation each modality.

Using this simple model, val_loss: 0.05, val_acc: 0.98

For the actual prediction, the accuracy is much higher.
Becuase I take multipl slices from the same scan, and for each slice, the prediction score is already high.
The program will make a decision based on the average of all the slices in a scan to determine modality.
