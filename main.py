import Exploring_datasets as ExplData
import Data_preprocessing as DataPre
import BERT_embedding as BertEmb
import Linear_and_Naive_Bayes as LNBa
import Neural_Network as NN
import numpy as np
import argparse

filename_ted = 'ted_main.csv'
train_set,valid_set,test_set = ExplData.data_import_ted(filename_ted)

#filename_yt = 'US_youtube_trending_data.csv'
#train_set_yt,valid_set_yt,test_set_yt = data_import_youtube(filename)

# ##1. EXPLORING THE DATASET
# ExplData.get_stats_and_plots_ted(train_set, valid_set, test_set)
# ExplData.vanilla_linear_regression_ted(train_set, valid_set)

##2. PREPROCESSING OF THE DATASET
train_names = train_set['title']
data_type = 'title'
tokenizing = False
dictionary = DataPre.create_dictionary(train_names,data_type, tokenizing)
print('Size of dictionary: ', len(dictionary))
train_matrix = DataPre.transform_text(train_names, dictionary,data_type, tokenizing)

# ##3. BERT EMBEDDING
# embedded_names = []
# train_names = train_set['title']
# for i in range(len(train_names)):
#     text = train_names.values[i]
#     embedded_names.append(BertEmb.get_embedding(text))

##4. LINEAR MODEL
number_views = train_set['views'].values
linear = LNBa.LinearModel()
linear.fit(train_matrix,number_views)
val_names = valid_set['title']
val_matrix = DataPre.transform_text(val_names, dictionary,data_type, tokenizing)

prediction = linear.predict(val_matrix)

##5. NAIVE-BAYES MODEL
number_views = train_set['views'].values
number_comments = train_set['comments'].values
naive_bayes_model_views = LNBa.fit_naive_bayes(train_matrix, number_views)
naive_bayes_model_comments = LNBa.fit_naive_bayes(train_matrix, number_comments)

top_5_words = LNBa.get_top_words_naive_bayes(5, naive_bayes_model_views, dictionary)
top_5_words_comments = LNBa.get_top_words_naive_bayes(5, naive_bayes_model_comments, dictionary)

print('The top 5 most successful words are: ', top_5_words)
print('Based on comments, the top 5 most successful words are: ', top_5_words_comments)

##6. NEURAL NETWORK
parser = argparse.ArgumentParser(description='Train a nn model.')
parser.add_argument('--num_epochs', type=int, default=30)
log_transfo = True

args = parser.parse_args()

if log_transfo:
    train_number_views = np.log(train_set['views'].values)
else:
    train_number_views = train_set['views'].values
train_data = train_matrix
train_labels = np.vstack(train_number_views)

if log_transfo:
    val_number_views = np.log(valid_set['views'].values)
else:
    val_number_views = valid_set['views'].values

val_data = val_matrix
val_labels = np.vstack(val_number_views)

test_names = test_set['title']
test_matrix = DataPre.transform_text(test_names, dictionary,data_type, tokenizing)
test_data = test_matrix

if log_transfo:
    test_number_views = np.log(test_set['views'].values)
else:
    test_number_views = test_set['views'].values
test_labels = np.vstack(test_number_views)

mean = np.mean(train_data)
std = np.std(train_data)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

all_data = {
    'train': train_data,
    'dev': val_data,
    'test': test_data
}

all_labels = {
    'train': train_labels,
    'dev': val_labels,
    'test': test_labels,
}
plot = True
baseline_acc = NN.run_train_test('baseline', all_data, all_labels, NN.backward_prop, args.num_epochs, log_transfo, plot)
reg_acc = NN.run_train_test('regularized', all_data, all_labels,
    lambda a, b, c, d, log_transfo: NN.backward_prop_regularized(a, b, c, d, log_transfo, reg=0.0001),
    args.num_epochs, log_transfo, plot)