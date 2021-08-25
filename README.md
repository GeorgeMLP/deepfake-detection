# DeepFake Detection

Codes for my homework project *DeepFake Detection*.

- Folder ```CelebA``` includes our source code and data for experiment on the CelebA dataset. Contents for each file are as follows:

  - ```celeba_low_1000.pkl``` are the power spectrum data for images in the dataset. You can also compute these data on your own by uncommenting corresponding codes in ```DeepFake Detection CelebA.py``` and change the paths into your local paths.
  - ```DeepFake Detection CelebA.py``` are codes for training and testing the classifier.
  - ```radialProfile.py``` is an auxiliary file.

  If you wish to compute the power spectrum on your own, download datasets at [DeepFakeDetection/dataset_celebA.7z](https://github.com/cc-hpc-itwm/DeepFakeDetection/blob/master/Experiments_CelebA/dataset_celebA.7z) and [Img - Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ) (download only ```img_align_celeba.zip``` in the second link) and change the paths into your local paths before running the codes. Otherwise, run the codes directly.

- Folder ```FaceForensics``` includes our source code and data for experiment on the FaceForensics dataset. Contents for each file are as follows:

  - ```DeepFake Detection FaceForensics.py``` are codes for training and testing the classifier.
  - ```radialProfile.py``` is an auxiliary file.
  - ```test_1000.pkl``` are the power spectrum data for images in the training set. You can also compute these data on your own by uncommenting corresponding codes in ```DeepFake Detection FaceForensics.py``` and change the paths into your local paths.
  - ```train_3200.pkl``` are the power spectrum data for images in the testing set. You can also compute these data on your own by uncommenting corresponding codes in ```DeepFake Detection FaceForensics.py``` and change the paths into your local paths.

  If you wish to compute the power spectrum on your own, download datasets at [prepro_deepFake.7z - Google Drive](https://drive.google.com/file/d/1rokPjCHe30mZBnk7J5j0MiIuUAuOqHoQ/view) and change the paths into your local paths before running the codes. Otherwise, run the codes directly.

- Folder ```Faces-HQ``` includes our source code and data for experiment on the Faces-HQ dataset. Contents for each file are as follows:

  - ```dataset_freq_1000.pkl``` are the power spectrum data for images in the dataset. You can also compute these data on your own by uncommenting corresponding codes in ```DeepFake Detection Faces-HQ.py``` and change the paths into your local paths.
  - ```DeepFake Detection Faces-HQ.py``` are codes for training and testing the classifier.
  - ```radialProfile.py``` is an auxiliary file.

  If you wish to compute the power spectrum on your own, download datasets at [faces.tar.gz - Google Drive](https://drive.google.com/file/d/1AqbGw82ueBP3fNNVCbXZgOPPFsh2uNXm/view) and change the paths into your local paths before running the codes. Otherwise, run the codes directly.

- ```Experiment Report.pdf``` is the experiment report of my homework.

- ```Poster.pdf``` is a poster for my project.