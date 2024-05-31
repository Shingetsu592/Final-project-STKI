import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.word as naw

# Download stopwords if not already downloaded
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords
stop_words = set(stopwords.words('indonesian'))

# Function to remove stopwords
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return filtered_text

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the CSV file
file_path = 'data600preprocessed.csv'
data = pd.read_csv(file_path)

# Remove stopwords from the text data
data['berita'] = data['berita'].apply(remove_stopwords)

# Define augmentation techniques and probabilities
synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='ind', aug_p=0.3)
swap_aug = naw.RandomWordAug(action="swap", aug_p=0.2)
delete_aug = naw.RandomWordAug(action="delete", aug_p=0.1)
aug_techniques = [synonym_aug, swap_aug, delete_aug]

# Function to augment text data
def augment_text(text, aug_techniques):
    augmented_texts = []
    for aug in aug_techniques:
        augmented_texts.extend(aug.augment(text))
    return augmented_texts

# Augment the text data
augmented_texts = []
for text in data['berita']:
    augmented_texts.extend(augment_text(text, aug_techniques))

# Create a new DataFrame with augmented data
augmented_data = pd.DataFrame({'berita': augmented_texts, 'tagging': np.repeat(data['tagging'].values, len(augmented_texts) // len(data))})

# Concatenate the original and augmented data
data = pd.concat([data, augmented_data], ignore_index=True)

# Extract text data
texts = data['berita'].tolist()

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModel.from_pretrained("indolem/indobert-base-uncased")

inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Apply the model to the tokenized dataset
with torch.no_grad():
    outputs = model(**inputs)

# Extract last hidden states1
last_hidden_states = outputs.last_hidden_state

# Print the shape of the last hidden states to verify the output
print(last_hidden_states.shape)

# Use the [CLS] token representation for each sequence as the feature vector
cls_embeddings = last_hidden_states[:, 0, :].numpy()

# Extract labels
labels = data['tagging'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(cls_embeddings, labels, test_size=0.2, random_state=42)

# Initialize individual classifiers
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
grad_boost = GradientBoostingClassifier()
svc = SVC(probability=True)
knn = KNeighborsClassifier()
mlp = MLPClassifier(max_iter=1000)

# Initialize the Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('rf', random_forest),
        ('gb', grad_boost),
        ('svc', svc),
        ('knn', knn),
        ('mlp', mlp)
    ],
    voting='soft'  # Use 'hard' for majority voting or 'soft' for weighted voting
)

# Function to evaluate a classifier
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Evaluate individual classifiers
classifiers = {
    "Logistic Regression": log_reg,
    "Random Forest": random_forest,
    "Gradient Boosting": grad_boost,
    "SVM": svc,
    "KNN": knn,
    "MLP": mlp
}

for name, clf in classifiers.items():
    print(f"Evaluating {name}...")
    accuracy, report = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("\n" + "="*80 + "\n")

# Evaluate the Voting Classifier
print("Evaluating Voting Classifier...")
accuracy, report = evaluate_classifier(voting_clf, X_train, X_test, y_train, y_test)
print(f"Voting Classifier Accuracy: {accuracy}")
print("Voting Classifier Classification Report:")
print(report)