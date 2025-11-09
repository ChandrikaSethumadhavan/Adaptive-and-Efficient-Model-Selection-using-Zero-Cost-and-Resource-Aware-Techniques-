# Adaptive-and-Efficient-Model-Selection-using-Zero-Cost-and-Resource-Aware-Techniques

## ⚙️ Technical Details

### 🧠 Model Architecture & 🧪 Trials and Experiments
<p align="center">
  <img src="https://github.com/user-attachments/assets/d96e7241-9876-4c1d-9c21-fd2dcb35d7e3" alt="Model Architecture" width="45%"/>
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/8337acfa-5c85-4c66-8306-f17699f82ed2" alt="Trials and Experiments" width="45%"/>
</p>

### 📊 Results on Multiple Datasets & Accuracy Across Trials
<p align="center">
  <img src="https://github.com/user-attachments/assets/139dc7c6-9fc8-444d-88ba-f6defc89886e" alt="Results on Datasets" width="45%"/>
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/dae433ca-95a4-4062-9549-5be3b7707298" alt="Accuracy Across Trials" width="45%"/>
</p>

### 🌍 Validation Accuracy & Carbon Footprint Comparison
<p align="center">
  <img src="https://github.com/user-attachments/assets/9b55b9ad-4d68-47f9-94c6-97e09683b3f1" alt="Validation Accuracy and Carbon Footprint" width="60%"/>
</p>



## Project Setup & Installation

Before running the code, set up your environment.

### Using `venv`

```bash
python3 -m venv automl-vision-env
source automl-vision-env/bin/activate
```

### Using `conda`

```bash
conda create -n automl-vision-env python=3.11
conda activate automl-vision-env
```

### Install Dependencies

Install the repo in editable mode:

```bash
pip install -e .
```
We have used all these modules in our project.
* `torchvision`
* `pandas`
* `scikit-learn`
* `numpy<2.0`
* `IPython`
* `optuna`
* `codecarbon`
As you can see we have added codecarbon and optuna to the requirements. If the install command does not work, You can install it manually by the following commands.

```bash
pip install codecarbon
pip install optuna
```

Test the install:

```bash
python -c "import automl"
```

---

## Code Structure

Our codebase is organized as follows:

```
src/
└── automl/
    ├── dac.py
    ├── model.py
    ├── run.py
    ├── training.py
    ├── utils.py
    ├── vision_datasets.py
    └── Zero_cost.py
```

* `run.py` — Main entry point for running Hyperparameter tunning, training and prediction.
* `training.py`, `model.py`, `utils.py` — Core modules implementing the AutoML logic.
* `dac.py`, `Zero_cost.py` — Additional search space and zero-cost proxy components.

We used both **Kaggle (P100 GPU)** and **Google Colab (T4 GPU)** for training and evaluation. It approximately took us around 3 hours to get the prediction on skin cancer dataset. The file is saved as
final_test_preds.npy

---

## Running Training & Predictions

You can run the pipeline like this to get the final_test_preds.npy :

```bash
python src/automl/run.py --dataset skin_cancer --n-trials 10 --seed 42  --carbon-budget 0.15 --enable-carbon-manager
```
Note: The carbon budget is customizable.

This project runs smoothly on **Kaggle** and **Google Colab**, GPU monitoring and carbon tracking are enabled with no extra steps required.

If you're running the pipeline on your **local machine**, you might notice a prompt asking for your password or administrator (sudo) permissions. This is because **CodeCarbon** may try to access system-level resources to monitor GPU and power usage.

### Model Performance Across Seeds (10 Trials Each)

| Dataset     | Metric                      | Seed 6        | Seed 42       | Seed 92       |
|-------------|-----------------------------|---------------|---------------|---------------|
| **flowers** | accuracy                    | 0.944         | 0.933         | 0.970         |
|             | F1                          | 0.938         | 0.931         | 0.967         |
| **emotions**| accuracy                    | 0.684         | 0.646         | 0.677         |
|             | F1                          | 0.668         | 0.636         | 0.662         |
| **fashion** | accuracy                    | 0.945         | 0.933         | 0.938         |
|             | F1                          | 0.945         | 0.933         | 0.937         |
| **skincancer** | accuracy                 | N/A           | **0.86**      | N/A           |
|             |     F1                      | N/A           | **0.79**      | N/A           |

### Final Metrics on the test dataset 
`final_test_preds.npy` results on Skin Cancer
- Accuracy: **0.8626**
- Precision - Micro: **0.8004**
- F1 - Micro: **0.7961**





Thanks to the University of Freiburg for helping shape this project with the AutoML course.


About
A resource-aware AutoML pipeline for deep learning on vision dataset

