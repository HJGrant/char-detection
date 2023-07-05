import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import io
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#Variables
image = [None] * 5
image_scaled = [None] * 5
predicted_label = [None] * 5
confidence = [None] * 5

#Charger les données 
mnist = load_digits()
X, y = mnist["data"], mnist["target"]

#Diviser l'ensemble de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#prétraitement des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# S'entrainer un ML model avec sklearn
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, random_state=42)
mlp.fit(X_train_scaled, y_train)


#definir les racines des images
image_paths = [
    "./mnist_images/"+str(30)+"_"+str(0)+".png",
    "./mnist_images/"+str(26)+"_"+str(6)+".png",
    "./mnist_images/"+str(42)+"_"+str(1)+".png",
    "./mnist_images/"+str(31)+"_"+str(9)+".png",
    "./mnist_images/"+str(35)+"_"+str(5)+".png"
]

for i in range(5):
    #Definir les racines des images
    image[i] = io.imread(image_paths[i])
    image[i] = image[i].reshape(1, -1) 
    image_scaled[i] = scaler.transform(image[i].astype(np.float64))

    #Valeurs de prediction et confiance
    predicted_label[i] = mlp.predict(image_scaled[i])
    confidence[i] = np.max(mlp.predict_proba(image_scaled[i]))


captions = [
    "Prédiction: "+str(predicted_label[0])+". Confiance: "+str(round(confidence[0],4)),
    "Prédiction: "+str(predicted_label[1])+". Confiance: "+str(round(confidence[1],4)),
    "Prédiction: "+str(predicted_label[2])+". Confiance: "+str(round(confidence[2],4)),
    "Prédiction: "+str(predicted_label[3])+". Confiance: "+str(round(confidence[3],4)),
    "Prédiction: "+str(predicted_label[4])+". Confiance: "+str(round(confidence[4],4))
]

font_size = 10

fig, axes = plt.subplots(1, 5, figsize=(15, 3))


for i, (image_path, caption) in enumerate(zip(image_paths, captions)):

    image = mpimg.imread(image_path)

    axes[i].imshow(image)
    axes[i].set_title(caption, fontsize=font_size)
    axes[i].axis('off')


plt.subplots_adjust(wspace=0.3)


plt.show()