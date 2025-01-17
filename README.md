 SMS Spam Detection

# SMS Spam Detection

This project aims to develop a machine-learning model to detect spam messages in SMS text data. It utilizes natural language processing (NLP) techniques and a supervised learning algorithm to classify SMS messages as either spam or non-spam (ham).

## Dataset

The dataset used for this project is the "SMS Spam Collection" from the UCI Machine Learning Repository. It contains a collection of 5,574 SMS messages, labeled as spam or ham. The dataset can be downloaded from [link to dataset] (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

The dataset file (`sms_spam_dataset.csv`) contains two columns:
- `label`: Indicates whether the message is spam (1) or ham (0).
- `text`: The actual text content of the SMS message.


## Results

The trained model achieved an **accuracy of 97.10 %** and **Precision is 100 %** on the test set and performed well in terms of precision, recall, and F1-score.

| Metric     | Score |
|------------|-------|
| Accuracy   | 97.10 %   |
| Precision | 100 %   |
| Recall     | 76.19 %   |
| F1-score   | 86.49 %   |


