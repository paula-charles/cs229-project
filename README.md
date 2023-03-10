# cs229-project

This is the code for the CS229 project "Finding the best TED Talk title", created by
Paula Charles and Marc Loning.

The file that we run is main.py. It calls all the other python files, which are composed
of functions and classes:
1. Exploring_datasets.py explores the 2 datasets, splits them in training/validation/test
datasets and tries to find interesting features.

2. Data_preprocessing.py pre-processes the data: it turns the titles into some inputs that
are read by our models. We can choose to split them into words or tokens (see paper for
explanation).

3. BERT_embedding.py: it turns the tokens into re-trained language model word representations.
This file is not used in our models, as we did not manage to include it into the neural
network, but is present for reference.

4. Linear_and_Naive_Bayes.py implements the linear model and the Naive Bayes one. It takes as
input the processed data.

5. Neural_Network.py implements the neural network, without and with regularization.

The two datasets that we use are:
- The TED talks dataset: ted_main.csv
- The Youtube videos dataset: US_youtube_trending_data.csv
