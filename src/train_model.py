import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# 1. CSV read karo
data = pd.read_csv('sign_data.csv')

# 2. X aur y split
X = data.drop('class', axis=1)
y = data['class']

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train karo SVM
model = SVC()
model.fit(X_train, y_train)

# 5. Accuracy dekho
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 6. Save karo model
pickle.dump(model, open('sign_model.pkl', 'wb'))
print("Model saved as sign_model.pkl")
