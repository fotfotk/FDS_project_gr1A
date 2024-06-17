# FDS project group 1A - Parkinson's dataset
## Project information and structure
The project is centred around binary classification of Parkinson's disease. It makes use of the Oxford Parkinson's Disease Detection Dataset, available on the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/174/parkinsons). This data should be downloaded prior to executing any scripts. Three models are used to achieve this task, namely Random Forests (with and without lasso regression), k-Nearest Neighbours, and Logistic regression. 
The project is structured as follows:
```bash
.
├── code/
│   ├── model_RF.py
│   ├── model_RF_with_LR.py
│   ├── knn_script.py
│   ├── logit_functions.py
│   ├── logit_script.py
├── data/
│   ├── parkinsons.data
├── output/
│   ├── knn/
│   ├── logit/
│   ├── randomforest/
└── README.md
```
## Execution instructions
Each model script is executed separately (the order does not matter). The logistic regression model makes use of helper functions defined in a separate file, ```logit_functions.py```, stored in the same directory as the script. All models make use of relative paths for extracting and storing data. If the project structure is changed, this should also be changed in each file.
