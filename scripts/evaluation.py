import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
model = tf.keras.models.load_model('models01')
test_data = np.load("eval_a_X.npy")
true_labels = np.argmax(np.load("y_validation_data.npy"),axis=1)

# true_labels = to_categorical(true_labels)
new_X_test_data = []

for i in range(test_data.shape[0]):
    scaler = StandardScaler()
    X_test_scaled = test_data[i][:3,:]
    X_test_scaled = np.append(X_test_scaled, test_data[i][4:5,:], axis=0)
    X_test_scaled = np.transpose(scaler.fit_transform(X_test_scaled))
    new_X_test_data.append(X_test_scaled)
new_X_test_data = np.array(new_X_test_data)
predictions = model.predict(new_X_test_data)

predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)
np.save("eval_a_y.npy", predicted_labels)
exit()
print(true_labels)
from sklearn.metrics import accuracy_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import numpy as np

accuracy = accuracy_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels, average='macro')  # Sensitivity
f1 = f1_score(true_labels, predicted_labels, average='macro')

cm = confusion_matrix(true_labels, predicted_labels)
specificity = np.zeros(cm.shape[0])
for i in range(cm.shape[0]):
    tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
    fn = np.sum(cm[i, :]) - cm[i, i]
    specificity[i] = tn / (tn + fn) if (tn + fn) != 0 else 0
# average_specificity = np.mean(specificity)

kappa = cohen_kappa_score(true_labels, predicted_labels)
import matplotlib.pyplot as plt
import seaborn as sns

class_names = ['W','1','2','3','4','R']
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("confusion matrix.png")
print(f"Accuracy: {accuracy}")
print(f"Sensitivity (Recall): {recall}")
print(f"F1 Score: {f1}")
print(f"Specificity per class: {specificity}")
print(f"Kappa Coefficient: {kappa}")