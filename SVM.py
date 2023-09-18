import sys as sys_module
import os
import torch
import argparse
import Clean_text
from SVMCustomDataset import SVMCustomDataset
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import Feature_set

class SVMCustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]

        #lexical_density = Feature_set.calculate_lexical_density_with_pos(text)
        #feature = lexical_density

        #sttr = Feature_set.calculate_sttr(text)
        #feature = sttr

        #average_sentence_length = Feature_set.calculate_average_sentence_length(text)
        #feature = average_sentence_length

        #lexical_function_word_ratio = Feature_set.caculate_lexical_function_word_ratio(text)
        #feature = lexical_function_word_ratio
 
        #passive_frequency = Feature_set.calculate_passive_voice_frequency_with_pos(text)
        #feature = passive_frequency
         
        top_words_frequency = Feature_set.calculate_top_words_frequency(text, Feature_set.top_words)
        feature = top_words_frequency
        
        #cohesive_markers_frequency = Feature_set.calculate_cohesive_markers_frequency(text, Feature_set.cohesive_markers)
        #feature = torch.tensor(cohesive_markers_frequency, dtype=torch.float32)

        #function_words_frequency = Feature_set.calculate_function_words_frequency(text, Feature_set.function_words)
        #feature = function_words_frequency

        #pronouns_frequency = Feature_set.caculate_pronouns_frequency(text, Feature_set.chinese_pronouns)
        #feature = pronouns_frequency

        #pos_bigrams_feature_vector = Feature_set.calculate_pos_bigrams_feature(text, Feature_set.top_bigrams)
        #feature = torch.tensor(pos_bigrams_feature_vector, dtype=torch.float32)

        #pos_triples_feature_vector = Feature_set.calculate_pos_triples_feature(text, Feature_set.top_triples)
        #feature = pos_triples_feature_vector
        
        label = label

        return feature, label
    
class CustomDataset_Positional_token_frequency(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]

        positional_token_frequency = Feature_set.caculate_positional_token_frequency(text)
        feature = torch.tensor(positional_token_frequency, dtype=torch.float32)
        
        # 将标签转化为 Tensor 格式
        label = torch.tensor(label, dtype=torch.int64)
        
        return feature, label

svm_custom_dataset = SVMCustomDataset(Clean_text.all_data)

features_list = []
labels_list = []

for sample in svm_custom_dataset:
    feature = sample[0]
    if isinstance(feature, float) or isinstance(feature, list):
        features_list.append([feature])
    else:
        features_list.append(feature.numpy())
    labels_list.append(sample[1])

X = np.array(features_list)
y = np.array(labels_list)

X = np.squeeze(X) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_accuracies = []
test_accuracies = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)

    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

avg_train_accuracy = np.mean(train_accuracies)
avg_test_accuracy = np.mean(test_accuracies)

print('Average Train Accuracy:', avg_train_accuracy)
print('Average Test Accuracy:', avg_test_accuracy)
