# CSEN 240 Term Project

Train a clasification model on radiology images to diagnose Knee Osteopenia and Osteoporosis.

Objective: Change hyper paremeters trying to improve the model's f1 scores.

## Changes

|Python File|Change Description|Log File|F1 Scores|
|-----------|------------------|--------|--------|
|[knee-osteo.py](/TermProject/knee-osteo.py)|Base implementation (no change)|[knee_base-161750.out](/TermProject/logs/knee_base-161750.out)|0.69,0.74,0.72|
|[knee-osteo-l2-01.py](/TermProject/knee-osteo-l2-01.py)|L2 regularization with 0.01 penalty|[knee_base-162715.out](/TermProject/logs/knee_base-162715.out)|0.78,0.72,0.73|
|[knee-osteo-l2-001.py](/TermProject/knee-osteo-l2-001.py)|L2 regularization with 0.001 penalty|[knee_base-162718.out](/TermProject/logs/knee_base-162718.out)|0.81,0.74,0.73|
|[knee-osteo-l2-0005.py](/TermProject/knee-osteo-l2-0005.py)|L2 regularization with 0.0005 penalty|[knee_base-162836.out](/TermProject/logs/knee_base-162836.out)|0.80,0.76,0.70|
|[knee-osteo-l2-0001.py](/TermProject/knee-osteo-l2-0001.py)|L2 regularization with 0.0001 penalty|[knee_base-162837.out](/TermProject/logs/knee_base-162837.out)|0.74,0.76,0.69|
|[knee-osteo_lr_1e-5.py](/TermProject/knee-osteo_lr_1e-5.py)|Set learning rate to 1e-5|[knee_base-162838.out](/TermProject/logs/knee_base-162838.out)|0.74,0.75,0.78|
|[knee-osteo_lr_5e-5.py](/TermProject/knee-osteo_lr_5e-5.py)|Set learning rate to 5e-5|[knee_base-162839.out](/TermProject/logs/knee_base-162839.out)|0.67,0.55,0.65|
|[knee-osteo-l2-001_lr_1e-5.py](/TermProject/knee-osteo-l2-001_lr_1e-5.py)|L2 reg. 0.001 penalty + LR = 1e-5|[knee_base-162840.out](/TermProject/logs/knee_base-162840.out)|0.71,0.73,0.65|
|[knee-osteo_lr_2e-5.py](/TermProject/knee-osteo_lr_2e-5.py)|Set learning rate to 2e-5|[knee_base-162864.out](/TermProject/logs/knee_base-162864.out)|0.70,0.71,0.71|
|[knee-osteo_lr_3e-5.py](/TermProject/knee-osteo_lr_3e-5.py)|Set learning rate to 3e-5|[knee_base-162865.out](/TermProject/logs/knee_base-162865.out)|0.73,0.77,0.68|
|[knee-osteo_lr_7_5e-5.py](/TermProject/knee-osteo_lr_7_5e-5.py)|Set learning rate to 7.5e-5|[knee_base-162866.out](/TermProject/logs/knee_base-162866.out)|0.71,0.76,0.73|
|[knee-osteo_lr_8e-5.py](/TermProject/knee-osteo_lr_8e-5.py)|Set learning rate to 8e-5|[knee_base-162867.out](/TermProject/logs/knee_base-162867.out)|0.73,0.75,0.69|
|[knee-osteo_lr_9e-5.py](/TermProject/knee-osteo_lr_9e-5.py)|Set learning rate to 9e-5|[knee_base-162868.out](/TermProject/logs/knee_base-162868.out)|0.78,0.76,0.70|
|[knee-osteo-l2-1e-3_lr_9e-5.py](/TermProject/knee-osteo-l2-1e-3_lr_9e-5.py)|L2 reg. 1e-3 penalty + LR = 9e-5|[knee_base-162965.out](/TermProject/logs/knee_base-162965.out)|-|
|[knee-osteo-l2-1e-3_lr_9e-5_dense_layer.py](/TermProject/knee-osteo-l2-1e-3_lr_9e-5_dense_layer.py)|L2 reg. 1e-3 penalty + LR = 9e-5 + 1 extra dense layer|[knee_base-162967.out](/TermProject/logs/knee_base-162967.out)|-|
|[knee-osteo-l2-1e-3_lr_9e-5_dense_layer_conv.py](/TermProject/knee-osteo-l2-1e-3_lr_9e-5_dense_layer_conv.py)|L2 reg. 1e-3 penalty + LR = 9e-5 + 1 extra dense layer + Convolutional layer|[knee_base-162970.out](/TermProject/logs/knee_base-162970.out)|-|

### Tested Learning Rates

Planning on testing learning rates. Listed in ascending order:

| Scientific Notation | Decimal Value | F1 Scores    |
|---------------------|---------------|--------------|
|1e-5                 |0.00001        |0.67,0.55,0.65|
|2e-5                 |0.00002        |0.70,0.71,0.71|
|3e-5                 |0.00003        |0.73,0.77,0.68|
|5e-5                 |0.00005        |0.67,0.55,0.65|
|7.5e-5               |0.000075       |0.71,0.76,0.73|
|8e-5                 |0.00008        |0.73,0.75,0.69|
|9e-5                 |0.00009        |0.78,0.76,0.70|
|1e-4                 |0.0001         |0.69,0.74,0.72|
