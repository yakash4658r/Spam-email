# Step 1 Import Libraries
import pandas as pd
#Meaning: Data va table format-la handle panna pandas use panrom.

import numpy as np
#Meaning: Maths calculation fast-a panna numpy use aagum.

from sklearn.model_selection import train_test_split
#Meaning: Namma kitta irukka data-va Training matrum Testing nu rendaa pirikka idhu use aagum.

from sklearn.feature_extraction.text import TfidfVectorizer
#Meaning: Computer-ku English words puriyadhu, numbers dhaan puriyum. So, text-a numbers-a maatha indha tool use panrom.

from sklearn.neural_network import MLPClassifier
#Meaning: Idhu dhaan namma Artificial Neural Network (ANN) model. Idhu dhaan Spam kandupidikka pogudhu.

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#Meaning: Model evalo correct-a vela seiyudhu nu check panna (Accuracy, Report) use panrom.

import seaborn as sns
import matplotlib.pyplot as plt
#Meaning: Graph matrum charts varaiya indha libraries use aagum.

import re
#Meaning: Text cleaning panna (like removing special symbols) use aagum.

import warnings
warnings.filterwarnings("ignore")
#Meaning: Code run aagum podhu theva illadha warning messages vara koodadhu nu off panrom.



# Step 2 Load Dataset

dataset = pd.read_csv("spam.csv", encoding="latin-1")
#Meaning: spam.csv ngra file ah open panni read panrom.

dataset = dataset[['v1', 'v2']]
#Meaning: Andha file-la neraya columns irukkalam, aana namaku v1 (Label) matrum v2 (Message) matum podhum.

dataset.columns = ['label', 'message']
#Meaning: Puriyura madhiri column names-a 'label' nu 'message' nu mathurom.

dataset['label'] = dataset['label'].map({'ham': 0, 'spam': 1})
#Meaning: Computer-ku 'ham', 'spam' puriyadhu. So, Ham (Not Spam) = 0 nnu Spam = 1 nnu number-a mathurom.

print("Dataset sample:")
print(dataset.head(), "\n")
#Meaning: Data epdi iruku nu paakka modhal 5 rows print panrom.


# Step 3: Text Cleaning and Preprocessing

def clean_text(text):
#Meaning: Oru pudhu function create panrom, idhu text-a clean pannum.
    text = text.lower()
#Meaning: Ella letters-ayum chinna ezhutha (lowercase) mathurom.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#Meaning: A-Z, numbers thavira vera edhavadhu special characters (!, @, #) irundha adha remove pannu.
    return text
#Meaning: Clean panna text-a thiruppi kudu.


dataset['clean_message'] = dataset['message'].apply(clean_text)
#Meaning: Mela ezhudhunna function-a ella message-kum apply panni, pudhu column-la store panrom.

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
#Meaning: Text-a number-a maatha ready aagurom. stop_words='english' na "is, the, a" madhiri mukkiyam illadha words-a remove pannidum. Top 3000 words mattum eduthukum.
X = vectorizer.fit_transform(dataset['clean_message'])
#Meaning: Ippo messages ellam numbers-a (vectors) maariduchu. Idhu dhaan Input (X).
y = dataset['label']
#Meaning: Idhu dhaan Answer Key (Target) using 0 or 1.


print("TF-IDF shape:", X.shape, "\n")

# Step 4 Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Meaning: Mottha data-la 80% Padikka (Train) matrum 20% Test panna pirikirom.

print("Train size:", X_train.shape)
print("Test size:", X_test.shape, "\n")

# Step 5: Build ANN Model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64),    
#rain-la rendu layers veikurom (First layer 128 neurons, second 64).                                
                    activation='relu',
#Oru math function, idhu data-va process panna help pannum.
                    solver='adam',
                    max_iter=20,
#20 times data-va paathu padikum.
                    random_state=42)

print("Training model... please wait 👀")
mlp.fit(X_train, y_train)
#Meaning: TRAINING START! Model ippo training data-va vechu padichitu iruku (Ham edhu, Spam edhu nu kathukudhu).
print("✅ Model training completed!\n")

# Step 6: Evaluate Model
y_pred = mlp.predict(X_test)
#Meaning: Test data-va kuduthu, "Idhu Spam ah illaya?" nu model kitta kekkurom.
acc = accuracy_score(y_test, y_pred)
#Meaning: Model sonna badhilum (y_pred), unmaiyana badhilum (y_test) onna irukka nu check panni mark podurom.
print("📊 Accuracy:", round(acc * 100, 2), "%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
#Meaning: Detail-a report generate panrom (Precision, Recall etc).


# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Meaning: Evalo correct, evalo thappu nu oru matrix create panrom.

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#Meaning: Andha matrix-a oru color diagram (Heatmap) ah varairom.

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


