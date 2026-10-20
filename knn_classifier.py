
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


iris = load_iris()
X, y = iris.data, iris.target
print("Samples:", X.shape[0], "| Features:", X.shape[1], "| Classes:", len(iris.target_names))
print("Features:", iris.feature_names)
print("Classes:", list(iris.target_names))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test  = scaler.transform(X_test)        


model = KNeighborsClassifier(n_neighbors=5)   
model.fit(X_train, y_train)                    
predictions = model.predict(X_test)            


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nF1 Score (weighted):", round(f1_score(y_test, predictions, average="weighted"), 4))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))
