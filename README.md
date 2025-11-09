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








Thanks to the University of Freiburg for helping shape this project with the AutoML course.

The aim of this repo is to provide a minimal installable template to help you get up and running.

For test results on _final dataset_ refer [here](#Running-auto-evaluation-on-test-dataset).

## Installation

To install the repository, first create an environment of your choice and activate it. 

For example, using `venv`:

You can change the python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-vision-env
source automl-vision-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-vision-env python=3.11
conda activate automl-vision-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:

```bash
python -c "import automl"
```

We make no restrictions on the python library or version you use, but we recommend using python 3.8 or higher.

## Code

We provide the following:

* `run.py`: A script that trains an _AutoML-System_ on the training split `dataset_train` of a given dataset and then
  generates predictions for the test split `dataset_test`, saving those predictions to a file. For the training
  datasets, the test splits will contain the ground truth labels, but for the test dataset which we provide later the
  labels of the test split will not be available. You will be expected to generate these labels yourself and submit
  them to us through GitHub classrooms.

* `src/automl`: This is a python package that will be installed above and contain your source code for whatever
  system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

**You are completely free to modify, install new libraries, make changes and in general do whatever you want with the
code.** The only requirement for the exam will be that you can generate predictions for the test splits of our datasets
in a `.npy` file that we can then use to give you a test score through GitHub classrooms.


## Data

We selected three different vision datasets which you can use to develop your AutoML system and we will provide you with
a test dataset to evaluate your system at a later point in time. The datasets can be automatically downloaded by the
respective dataset classes in `./src/automl/datasets.py`. The datasets are: _fashion_, _flowers_, and _emotions_.

If there are any problems downloading the datasets, you can download them manually:

- Practice datasets : https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase1.zip
- Final test dataset : https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase2.zip

After downloading, unzip them and place the contents in the `/data` folder.

The downloaded datasets will have the following structure:
```bash
./data
├── fashion
│   ├── images_test
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   ├── images_train
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   ├── description.md
│   ├── test.csv
│   └── train.csv
├── emotions
    ...
...
```
Feel free to explore the images and the `description.md` files to get a better understanding of the datasets.
The following table will provide you an overview of their characteristics and also a reference value for the 
accuracy that a naive AutoML system could achieve on these datasets:

| Dataset name | # Classes | # Train samples | # Test samples | # Channels | Resolution | Reference Accuracy |
|--------------|-----------|-----------------|----------------|------------|------------|--------------------|
| fashion      | 10        | 60,000          | 10,000         | 1          | 28x28      | 0.88               |
| flowers      | 102*      | 5732            | 2,457          | 3          | 512x512    | 0.55               |
| emotions     | 7         | 28709           | 7,178          | 1          | 48x48      | 0.40               |
| **skin_cancer**  |**7\***       | **7,010**          | **3,005**          | **3**          | **450x450**    | **0.71**               |

*classes are imbalanced

The final test dataset is `skin_cancer`along with its class definition to the `datasets.py` file. 
The test dataset is in the same
format as the training datasets, but `test.csv` will only contain nan's for labels.



## Running an initial test

This will download the _fashion_ dataset into `./data`, train a dummy AutoML system and generate predictions for the test
split:

```bash 
python run.py --dataset fashion --seed 42 --output-path preds-42-fashion.npy
```

You are free to modify these files and command line arguments as you see fit.

## Running auto evaluation on test dataset

Only activates on push to the `test` branch. It is important to note that Github Classroom creates unrelated histories for the `main` branch and `test` branch, that is why you can not use `git merge main` from the `test` branch directly. There are many ways to move the changes from other branches (e.g. from the `main` branch) to the `test` branch even though the commit histories between the branches are unrelated. Here is a simple way:
```bash
# on some_branch (e.g. main) do:
git add data/exam_dataset/predictions.npy
git commit -m "Generated predictions for test data"
git checkout test
#now you should be in the test branch
git checkout some_branch -- data/exam_dataset/predictions.npy # only copies the data/exam_dataset/predictions.npy to the test branch and stages it, ready to be comitted
git status # ensure that your latest `.data/exam_dataset/predictions.npy` is staged
git commit -m "Generated predictions for test data, ready for evaluation"
git push
# wait for some time (few seconds) or monitor the web UI of Github to see if the job ran successfully
git pull 
# test scores will be downloaded under `.data/exam_dataset/test_out/` if the job ran successfully
```

Feel free to use any other command to move the prediction files from other branches with unrelated histories to the test branch (`rebase`,`merge some_branch_with_unrelated_history --allow-unrelated-histories`, `stash`...), **<span style="color:red">just make sure that there is nothing else inside `data/exam_dataset/` except for `predictions.npy` and the evaluation results that we push</span>**.

A summary of the evaluation workflow:
* To initialize auto-evaluation for the test data, checkout to the `test` branch.
* Make sure you have named the prediction file `predictions.npy` and placed it in the `data/exam_dataset/` directory in this branch.
* After pushing to it, the evaluation script will be automatically triggered.
* The results are also pushed to your repo (don't forget to `git pull`)
* If no new commits are pulled by `git pull`, check the errors in the Github's `Action` section (red cross inline, last commit message, test branch)

### <span style="color:red">Important: The dir `data/exam_dataset/` should contain only the `predictions.npy` file and the result files we push, nothing else</span>.

```bash
./data
└── exam_dataset
│   └── predictions.npy
│   └── test_out
│   │   ├── test_evaluation_output_2025-MM-DD_HH-mm-ss-ms
.   .   .
```

<span style="color:red"> **Note that any edits to the yaml workflow script are prohibited and monitored!** </span>







## Tips



* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
  predictions, etc, Ignore your virtual environment and any additional folders/files
  created by your IDE.

About
A resource-aware AutoML pipeline for deep learning on vision dataset

© 2025 GitHub, Inc.
Footer navigation
Term
