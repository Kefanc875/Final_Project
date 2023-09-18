import argparse
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import Feature_set
import Clean_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]

        #lexical_density = Feature_set.calculate_lexical_density_with_pos(text)
        #feature = torch.tensor([lexical_density], dtype=torch.float32)

        #sttr = Feature_set.calculate_sttr(text)
        #feature = torch.tensor([sttr], dtype=torch.float32)

        #lexical_function_word_ratio = Feature_set.caculate_lexical_function_word_ratio(text)
        #feature = torch.tensor([lexical_function_word_ratio], dtype=torch.float32)

        #average_sentence_length = Feature_set.calculate_average_sentence_length(text)
        #feature = torch.tensor([average_sentence_length], dtype=torch.float32)
 
        #passive_frequency = Feature_set.calculate_passive_voice_frequency_with_pos(text)
        #feature = torch.tensor([passive_frequency], dtype=torch.float32)
         
        #top_words_frequency = Feature_set.calculate_top_words_frequency(text, Feature_set.top_words)
        #feature = torch.tensor(top_words_frequency, dtype=torch.float32)
        
        #cohesive_markers_frequency = Feature_set.calculate_cohesive_markers_frequency(text, Feature_set.cohesive_markers)
        #feature = torch.tensor(cohesive_markers_frequency, dtype=torch.float32)
       
        #function_words_frequency = Feature_set.calculate_function_words_frequency(text, Feature_set.function_words)
        #feature = torch.tensor(function_words_frequency, dtype=torch.float32)

        #pronouns_frequency = Feature_set.caculate_pronouns_frequency(text, Feature_set.chinese_pronouns)
        #feature = torch.tensor(pronouns_frequency, dtype=torch.float32)

        #punctuation_frequency = Feature_set.caculate_punctuation_frequency(text, Feature_set.chinese_punctuation, Feature_set.target_punctuation)
        #feature = torch.tensor(punctuation_frequency, dtype=torch.float32)

        #pos_bigrams_feature_vector = Feature_set.calculate_pos_bigrams_feature(text, Feature_set.top_bigrams)
        #feature = torch.tensor(pos_bigrams_feature_vector, dtype=torch.float32)

        pos_triples_feature_vector = Feature_set.calculate_pos_triples_feature(text, Feature_set.top_triples)
        feature = torch.tensor(pos_triples_feature_vector, dtype=torch.float32)
        
        label = torch.tensor(label, dtype=torch.int64)
        
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
        
        label = torch.tensor(label, dtype=torch.int64)
        
        return feature, label

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def infer(model, dataset, device):
    model.eval()
    acc_num = 0.0
    with torch.no_grad():
        for data in dataset:
            datas, labels = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_y, labels.to(device)).sum().item()
    accuracy = acc_num / len(dataset)
    return accuracy

def train_and_validate(train_loader, validate_loader, model, optimizer, loss_function, device):
    model.train()
    acc_num = torch.zeros(1).to(device)
    sample_num = 0
    for datas in train_loader:
        data, label = datas
        label = label.squeeze(-1)
        sample_num += data.shape[0]

        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        outputs = model(data)
        pred_class = torch.max(outputs, dim=1)[1]
        acc_num += torch.eq(pred_class, label.to(device)).sum()

        loss = loss_function(outputs, label.to(device))
        loss.backward()
        optimizer.step()

    train_acc = acc_num.item() / sample_num

    # Validation
    val_accurate = infer(model=model, dataset=validate_loader, device=device)

    return train_acc, val_accurate

def main(args, custom_dataset):
    print(args)

    data_list = [(feature, label) for feature, label in custom_dataset]

    K = 5  
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    kfold_metrics = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(data_list, [label for _, label in data_list])):  
        print(f"Fold {fold + 1}/{K}:")

        # Create the model and load the weights
        model = NeuralNetwork(300, 12, 6, 12, 2)  # Instantiate model
        model.to(device)
        loss_function = nn.CrossEntropyLoss()  # Define loss function
        pg = [p for p in model.parameters() if p.requires_grad]  # Define model parameters
        optimizer = optim.Adam(pg, lr=args.lr)  # Define optimizer

        train_data = [custom_dataset[i] for i in train_indices]
        val_data = [custom_dataset[i] for i in val_indices]

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

        for epoch in range(args.epochs):
            model.train()
            acc_num = torch.zeros(1).to(device)
            sample_num = 0
            train_bar = tqdm(train_loader, file=sys.stdout, ncols=100, desc="train")
            for datas in train_bar:
                data, label = datas
                label = label.squeeze(-1)
                sample_num += data.shape[0]
                optimizer.zero_grad()
                data, label = data.to(device), label.to(device)
                outputs = model(data)
                pred_class = torch.max(outputs, dim=1)[1]
                acc_num += torch.eq(pred_class, label.to(device)).sum()

                loss = loss_function(outputs, label.to(device))
                loss.backward()
                optimizer.step()

                train_acc = acc_num.item() / sample_num
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)

        val_accuracy = infer(model, val_loader, device)
        kfold_metrics.append(val_accuracy)
        print(f"Validation Accuracy (Fold {fold + 1}/{K}): {val_accuracy:.3f}")

    average_accuracy = sum(kfold_metrics) / len(kfold_metrics)
    print(f"Average Validation Accuracy (K={K}): {average_accuracy:.3f}")

    print('Finished K-fold Cross Validation')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
    parser.add_argument('--epochs', type=int, default=20, help='the number of training epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='star learning rate')   
    parser.add_argument('--data_path', type=str, default="/mnt/d/Codes/GNN/NN/Iris_data.txt") 
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    custom_dataset = CustomDataset(Clean_text.all_data)
    #custom_dataset = CustomDataset_Positional_token_frequency(Clean_text.all_sentence)

    main(args, custom_dataset)


