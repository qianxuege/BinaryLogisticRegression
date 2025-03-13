## Description
This is a working Natural Language Processing (NLP) system that determines whether a restaurant review is positive or negative, using binary logistic regression.

### Examples of Data
```
1 i will never forget this single breakfast experience in mad...
0 the search for decent chinese takeout in madison continues ...
0 sorry but me julio fell way below the standard even for med...
1 so this is the kind of food that will kill you so there s t...
```

### Feature Engineering
`feature.py` takes in the raw input data and produces a real-valued vector for each training, validation, and test example.  
`glove_embeddings.txt` contains the GloVe embeddings of 6792 words, used for feature engineering.

### Binary Logistic Regression
`lr.py` takes in the real-valued vectors and trains a logistic regression model using stochastic gradient descent to predict whether each example is a positive or negative review.  
`nll.py` plots the average negative log-likelihood for the training and validation data sets after each of 1,000 epochs. The y-axis shows the negative log-likelihood and the x-axis shows the number of epochs. It demonstrates the effect of overfitting, in which as the number of epochs increase, the model is learning details from the training data that don’t generalize well to the validation data.

## Instructions to Run the Model
### feature.py
```
python feature.py \
largedata/train_large.tsv \
largedata/val_large.tsv \
largedata/test_large.tsv \
glove_embeddings.txt \
test_output/formatted_train_large.tsv \
test_output/formatted_val_large.tsv \
test_output/formatted_test_large.tsv
```
### lr.py
```
python lr.py \
test_output/formatted_train_large.tsv \
test_output/formatted_val_large.tsv \
test_output/formatted_test_large.tsv \
test_output/formatted_train_labels.txt \
test_output/formatted_test_labels.txt \
test_output/formatted_metrics.txt \
500 \
0.1
```
### nll.py
```
python nll.py \
test_output/formatted_train_large.tsv \
test_output/formatted_val_large.tsv \
test_output/formatted_test_large.tsv \
test_output/formatted_train_labels.txt \
test_output/formatted_test_labels.txt \
test_output/formatted_metrics.txt \
1000 \
0.1
```
