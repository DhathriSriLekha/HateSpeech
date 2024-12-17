import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # for saving the model

# Load the dataset
dataset = pd.read_csv('C:/Users/arive/OneDrive/Desktop/AIML/output_file.csv')

# Data preprocessing
dataset = dataset.dropna(subset=['tweet', 'class'])  # Drop rows with missing text or labels

# Check class distribution
print(f'Class distribution:\n{dataset["class"].value_counts()}')

# Define features and target
features = dataset['tweet']
target = dataset['class']

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Use TF-IDF Vectorizer to convert text to numerical representation
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training data and transform the validation data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Define the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train_tfidf, Y_train)

# Make predictions on the validation set
y_pred = dt_classifier.predict(X_val_tfidf)

# Evaluate the model
accuracy = accuracy_score(Y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(Y_val, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred))

# Save the trained model and TF-IDF vectorizer
joblib.dump(dt_classifier, r"C:/Users/arive/OneDrive/Desktop/AIML/project/model/hate_speech_model.pkl")
joblib.dump(tfidf_vectorizer, r"C:/Users/arive/OneDrive/Desktop/AIML/project/model/tfidf_vectorizer.pkl")
