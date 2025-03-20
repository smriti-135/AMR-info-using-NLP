import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from spacy.training import offsets_to_biluo_tags
from spacy.lang.en import English
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel

# Mount Google Drive
def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')

# Load Dataset
def load_data():
    data = pd.read_json("/content/drive/MyDrive/project_9_dataset.jsonl", lines=True)
    labels_data = pd.read_json("/content/drive/MyDrive/project_9_labels.json")
    return data, labels_data

data, labels_data = load_data()

# Step 1: Data Preprocessing
## Extracting Text and Labels

def extract_annotations(data, labels_data):
    text = []
    label = []
    for i in range(len(data)):
        if data['annotations'][i]:
            text.append(data['text'][i])
            start_offset, end_offset, label_name = [], [], []
            for ann in data['annotations'][i]:
                start_offset.append(ann['start_offset'])
                end_offset.append(ann['end_offset'])
                label_name.append(labels_data[labels_data['id'] == ann['label']]['text'].values[0])
            label.append(list(zip(start_offset, end_offset, label_name)))
        else:
            text.append(data['text'][i])
            label.append([("")])
    return text, label

text, label = extract_annotations(data, labels_data)

# Step 2: Convert to BILUO Tags
def convert_to_biluo(text, label):
    list_of_lines = []
    list_of_tags = []
    for i in range(len(text)):
        nlp = English()
        offsets = label[i]
        doc = nlp(text[i])
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text[i]), offsets)
            list_of_lines.append([token.text for token in doc])
            list_of_tags.append(tags)
        except:
            pass
    return list_of_lines, list_of_tags

list_of_lines, list_of_tags = convert_to_biluo(text, label)

# Step 3: Tokenization and Encoding using BioBERT

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

def tokenize_and_align_labels(list_of_lines, list_of_tags):
    tokenized, aligned_tags = [], []
    for i, sentence in enumerate(list_of_lines):
        tokenized_sentence = []
        tag_sequence = []
        for word, tag in zip(sentence, list_of_tags[i]):
            tokens = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokens)
            tag_sequence.extend([tag] + ['X'] * (len(tokens) - 1))
        tokenized.append(tokenized_sentence)
        aligned_tags.append(tag_sequence)
    return tokenized, aligned_tags

tokenized_sentences, aligned_tags = tokenize_and_align_labels(list_of_lines, list_of_tags)

# Step 4: Padding and Attention Mask
max_length = 318

def pad_sequences_and_create_mask(tokenized_sentences, aligned_tags):
    token_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sentences]
    padded_tokens = pad_sequences(token_ids, maxlen=max_length, padding='post')
    padded_tags = pad_sequences([[tag for tag in tags] for tags in aligned_tags], maxlen=max_length, padding='post', value=-1)
    attention_masks = [[1 if token != 0 else 0 for token in sent] for sent in padded_tokens]
    return padded_tokens, padded_tags, attention_masks

padded_tokens, padded_tags, attention_masks = pad_sequences_and_create_mask(tokenized_sentences, aligned_tags)

# Step 5: Model Training
def create_model():
    class BioBERTNER(nn.Module):
        def __init__(self):
            super(BioBERTNER, self).__init__()
            self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
            self.dropout = nn.Dropout(0.5)
            self.output = nn.Linear(self.bert.config.hidden_size, len(set(aligned_tags)))

        def forward(self, inputs, attention_mask):
            bert_output = self.bert(inputs, attention_mask=attention_mask)[0]
            logits = self.output(self.dropout(bert_output))
            return logits

    return BioBERTNER()

model = create_model()

def train_model(model, padded_tokens, padded_tags, attention_masks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_tokens, test_tokens, train_tags, test_tags, train_masks, test_masks = train_test_split(
        padded_tokens, padded_tags, attention_masks, test_size=0.3, shuffle=True
    )
    
    train_dataset = TensorDataset(torch.tensor(train_tokens), torch.tensor(train_tags), torch.tensor(train_masks))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(10):
        for batch in tqdm(train_loader):
            tokens, tags, masks = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(tokens, masks)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
            loss.backward()
            optimizer.step()

train_model(model, padded_tokens, padded_tags, attention_masks)

# Step 6: Model Evaluation
def evaluate_model(model, padded_tokens, padded_tags, attention_masks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_dataset = TensorDataset(torch.tensor(padded_tokens), torch.tensor(padded_tags), torch.tensor(attention_masks))
    test_loader = DataLoader(test_dataset, batch_size=32)
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            tokens, tags, masks = [t.to(device) for t in batch]
            outputs = model(tokens, masks)
            predictions.extend(outputs.argmax(dim=2).cpu().numpy())
            true_labels.extend(tags.cpu().numpy())
    print(classification_report(true_labels, predictions))

evaluate_model(model, padded_tokens, padded_tags, attention_masks)
