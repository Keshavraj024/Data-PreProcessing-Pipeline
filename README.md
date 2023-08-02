# Data Pre-Processing for Machine Learning Tasks

A Python script for data pre-processing tasks commonly used in machine learning projects. The script handles missing data, performs one-hot encoding of categorical data, encodes labels, splits the data into training and testing datasets, and feature scales the data using either normalization or standardization.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Keshavraj024/Data-PreProcessing-Pipeline.git
cd Data-PreProcessing-Pipeline
```

2. Place your dataset in the root directory of the project

3. Run the script to start the data pre-processing pipeline:

```bash
python3 data_preprocessing.py your_dataset.csv strategy split_percentage
```

Replace `your_dataset.csv` with the filename of your dataset. `strategy` can be one of the following:
   - "mean" (default): Fill missing values with the mean of the column.
   - "median": Fill missing values with the median of the column.
   - "most_frequent": Fill missing values with the most frequent value of the column.
   - "constant": Fill missing values with a constant value (you will be prompted to enter the value).

`split_percentage` is the percentage of the dataset to use for testing (default is 0.2)

4. The script will process the dataset, split it into training and testing sets, and perform the specified data pre-processing tasks

5. The pre-processed data will be saved to the following CSV files in the project root directory:
   - train_set_x.csv: Training features
   - train_set_y.csv: Training labels
   - test_set_x.csv: Testing features
   - test_set_y.csv: Testing labels

