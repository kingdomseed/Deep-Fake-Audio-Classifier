# Mini Project: Audio Deepfake Detection via Binary Classification

IMPORTANT: The data and scripts in this repo are final.

## 1. Overview

This project involves performing binary classification on audio data, to detect whether an audio is a deepfake or not. The dataset consists of pre-processed features extracted from audio files. Your objective is to develop a model that outputs prediction scores, evaluate its performance using the provided scripts, and participate in the course leaderboard.

> I have already trained and tested the model on a laptop using CPU only, so it should be also doable for you to train without the need of a GPU.

There will be two QA sessions per week (mainly by HiWis).

- Jan 27 (Tuesday)
- Jan 29 (Thursday)
- Feb 3 (Tuesday)
- Feb 6 (Friday)

Please follow the [ticket system](How%20to%20Ask%20Questions%3A%20The%20Ticket%20System.md) to ask questions by adding a "ticket" to the Github Issues.
Tickets will be first handled by HiWis, if they cannot answer, they will be forwarded to me.

## 2. Dataset Structure

You can download the data from the [link](https://zenodo.org/records/18342979)

It is a zip file, after unzipping, you will get the following directory structure:
### 2.1 Directory Layout

```text
data/
├── train/
│   ├── features.pkl
│   └── labels.pkl
├── dev/
│   ├── features.pkl
│   └── labels.pkl
└── test1/
    └── features.pkl

```

The data is provided in Pandas DataFrame format, saved as `.pkl` files. The features are `torch.Tensor` composed of three concatenated feature types: [lfcc, delta, delta-delta]

### 2.2 Data Specification

Each `features.pkl` file contains a Pandas DataFrame with the following attributes:

* `uttid`: Unique identifier for each audio sample.
* `features`: A `torch.Tensor` containing the 2-dimensional feature vector [`num_feature_dimensions`, `T`]

**2.2.1 Example: Loading and inspecting the data**

**features.pkl example:**

```python
import pandas as pd
import pickle

with open('data/test/features.pkl', 'rb') as f:
    features_df = pd.read_pickle(f)

print(features_df.head())
```

Example output:

```
       uttid                                           features
0  raw_31131  [[tensor(4.1947), tensor(4.6297), tensor(5.824...
1  raw_30910  [[tensor(-13.4282), tensor(-14.7175), tensor(-...
2  raw_49920  [[tensor(-10.1677), tensor(-14.4544), tensor(-...
3   raw_1785  [[tensor(-46.2565), tensor(-47.0128), tensor(-...
4  raw_11332  [[tensor(-12.9300), tensor(-16.4384), tensor(-...
```

- `raw_31131` is the utterance id
- `features` is a `torch.Tensor` of shape `[180, 321]` (180 time frames × 321 feature dimensions)

**2.2.2 labels.pkl example:**

```python
import pandas as pd

with open('data/test/labels.pkl', 'rb') as f:
    labels_df = pd.read_pickle(f)

print(labels_df.head())
```

Example output:

```
       uttid  label
0  raw_31131      1
1  raw_30910      0
2  raw_49920      1
3   raw_1785      0
4  raw_11332      1
```

- `raw_31131` is the utterance id
- `label` is a `int` of value `1` (bonafide) or `0` (deepfake)


## 3. Evaluation and Leaderboard

### 3.1 Metrics: Equal Error Rate (EER)

Equal Error Rate (EER) is a common metric for binary classification. It's the point where the false acceptance rate equals the false rejection rate.

Lower is better.
- EER = 50%, the model is random.
- EER = 0%, the model is perfect.
- EER = 100%, means perfectly wrong prediction. In this case, if you flip the predictions (e.g., multiply by -1), you will get the perfect performance. An EER close to 100% usually indicates that the labels were flipped during training.

The implementation of EER is provided in the `scripts/evaluation.py` file.

### 3.2 Prediction File Format

You must generate a `prediction.pkl` file containing a DataFrame with two attributes:

1. `uttid`: Matching the IDs in the feature set.
2. `predictions`: The raw output/probability from your model.

**prediction.pkl example:**

An example `prediction.pkl` file is provided in the `examples/` directory. You can load and inspect it as follows:

```python
import pandas as pd

with open('examples/prediction.pkl', 'rb') as f:
    prediction_df = pd.read_pickle(f)

print(prediction_df.head())
```

Example output:

```
       uttid  predictions
0  raw_31131         0.85
1  raw_30910         0.23
2  raw_49920         0.67
3   raw_1785         0.91
4  raw_11332         0.12
```

- `raw_31131` is the utterance id
- `predictions` is a `float` (it can be logits, probabilities, scores, etc.)

### 3.3 Evaluation Script

Use the provided `scripts/evaluation.py` to calculate the Equal Error Rate (EER). 
Use the provided `scripts/generate_submission.py` to generate the submission file for the leaderboard.

**Usage:**

```bash
python scripts/evaluation.py <prediction.pkl> <labels.pkl> 
python scripts/generate_submission.py test2/features.pkl <prediction.pkl> <Student_ID> <FirstName> <LastName> <Nickname>
```

* `Student_ID`: Your "st" number.
* `Nickname`: The name to be displayed on the public leaderboard.

This `scripts/generate_submission.py` script generates a file named `<student_id>-<first_name>-<last_name>-<nickname>.pkl`. **Do not rename this file**, as it may cause errors in automated processing.

### 3.4 Leaderboard Submission

Upload the generated `.pkl` file to **ILIAS**. Starting from Jan 29, the leaderboard will be updated daily by us. Only nicknames will be displayed publicly to maintain privacy; student IDs are used solely for verification.

## 4. Example Code

[demo.py](examples/demo.py) showcases how to structure your code for this task.
Notice that it is not runnable, it is just a example code.

You don't need to follow the code exactly. You are free to choose your own structure, models, losses, even data.
But you should make sure the prediction file is in the correct format, otherwise it can't be processed by the evaluation scripts.


## 5. Final Submission Requirements


On **Feb 6**, the final test set (features only, no labels) will be released.
1. Run inference on the new test set.
2. Generate the submission file using `scripts/generate_submission.py`.
3. Prepare a Markdown file based on the provided template **briefly** explaining your model architecture and methodology (less than 400 words).

Submit to the **Final Submission** Exercise on ILIAS (Will be available on Feb 6).

## 6. Bonus Points

You will get bonus points if you participate in the project (and of course, submit a non-randomly-generated prediction file). i.e., if your model gets a score slightly better than a random baseline, it is considered as a valid submission. A random baseline is a model that gets 50% of Equal Error Rate (EER). If you model gets a score < 45% EER, then it is valid. 

Bonus points will range from 0 to 6. We will assign bonus points based on EER, using a non-linear scaling from roungly 45% to 1%.

What does a full score (6.0) mean?
- Based on results from previous semesters, a score of 6 can improve a student's final grade by up to **two** levels. e.g., jump from 2.3 to 1.7 (2.3 -> 2.0 -> 1.7, in edge cases). More commonly, it improves the grade by one level, e.g., from 1.3 to 1.0.

## 7. Important Dates

| Event | Date |
| --- | --- |
| Project Start | January 27 |
| Leaderboard Open | January 29 | 
| Test Set Release | February 6 | 
| Submission Deadline | February 9  (8 PM) |

**Note:** The submission window closes strictly at 8 PM on Feb 9. Late submissions will not be processed. 
