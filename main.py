import Exploring_datasets as ExplData
import Data_preprocessing as DataPre
import BERT_embedding as BertEmb
import Naive_Bayes_model as NBa

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
tokenizing = True
dictionary = DataPre.create_dictionary(train_names,data_type, tokenizing)
print('Size of dictionary: ', len(dictionary))
train_matrix = DataPre.transform_text(train_names, dictionary,data_type, tokenizing)