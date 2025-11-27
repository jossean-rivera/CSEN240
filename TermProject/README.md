# CSEN 240 Term Project

Train a clasification model on radiology images to diagnose Knee Osteopenia and Osteoporosis.

Objective: Change hyper paremeters trying to improve 

## Changes

|Python File|Change Description|Log File|F1 Scores|
|-----------|------------------|--------|--------|
|[knee-osteo.py](/TermProject/knee-osteo.py)|Base implementation (no change)|[knee_base-161750.out](/TermProject/logs/knee_base-161750.out)|0.69,0.74,0.72|
|[knee-osteo-l2-01.py](/TermProject/knee-osteo-l2-01.py)|L2 regularization with 0.01 penalty|[knee_base-162715.out](/TermProject/logs/knee_base-162715.out)|0.78,0.72,0.73|
|[knee-osteo-l2-001.py](/TermProject/knee-osteo-l2-001.py)|L2 regularization with 0.001 penalty|[knee_base-162718.out](/TermProject/logs/knee_base-162718.out)|0.81,0.74,0.73|
