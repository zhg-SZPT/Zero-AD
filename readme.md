# Zero-AD: Zero-Sample Anomaly Detection for Industrial Products
The code for the paper "Zero-AD: Zero-Sample Anomaly Detection for Industrial Products".
## Requirements
python 3.8.10 <br>
pytorch 1.9.0 <br>
cuda 11.1 <br>
## Dataset Preparation
1.This article has provided a data set label in the .txt format. You only need to download the data set before training. And if you need our dataset, please contact our corresponding author and you will be required to sign a confidentiality agreement.
Corresponding author email : zhg2018@sina.com<br>
2.In the dataset_txt folder, contains "stain-2", "gap", "filamentous", "height mismatch", and "unknown" subfields. The code is based on "stain-2" as an example. The "unknown" file is only used in gan_test.py.<br>
## Train
1.train_c.py is the file of the first model of the training paper, and gan_train.py is the file of the second part of the training paper. <br>
2.The default parameters are used for BC Defects datasets, which can be run directly.<br>
3.If you need to modify the defect category, pay attention to modify the label parameters in the SiameseNetworkDataset and pred_label_process function. For example, to change the stain-2 to the gap.<br> 
img1_list [4] == '1' => img1_list [3] == '1'.<br>
preds_new = del_tensor_ele_n(preds_new, 2, 1) => preds_new = del_tensor_ele_n(preds_new, 3, 1)
  ```python
  stain-2 = 3
  gap = 4
  filamentous = 5
  height_mismatch = 6
  (Note: Several categories of label index during training)
  ```
4.After modifying the classes_path, you can run train_c.py or gan_train to start training. After training multiple epochs, the weights and their generated images will be saved in 'output' or 'GAN_output' folder.
## test
1.For the test_c.py file, if the defect category needs to be changed, modify the data set address and the pred_label_process function.<br>
2.For gan_test.py files, if the defect category is modified, the parameter unknown_type_index needs to be modified. At this time, the defect category index is shown below:
  ```python
  stain-2 = 2
  gap = 3
  filamentous = 4
  height_mismatch = 5
  ```
## Citation
## Acknowledgement
