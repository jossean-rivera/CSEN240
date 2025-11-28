# CSEN 240 Term Project

Train a clasification model on radiology images to diagnose Knee Osteopenia and Osteoporosis.

Objective: Change hyper paremeters trying to improve 

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
