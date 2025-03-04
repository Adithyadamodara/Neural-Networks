import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime 
import lime.lime_tabular

# Sample dataset to train model
data = {
    'Area' : [1200,1500,800,950,1800],
    'Bedrooms' : [2,3,1,1,4],
    'Location' : [1,2,1,2,3],
    'Price' : [300000,350000,200000,220000,400000]
}

df = pd.DataFrame(data)

# Split into input(X) and output(Y)
x = df[['Area','Bedrooms','Location']]
y = df['Price']

# Split into training and testing sets
X_train, X_test,Y_train, Y_test = train_test_split(x,y, test_size=0.2,random_state=42) 

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple nerual network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    keras.layers.Dense(5, activation='relu'),  # Hidden layer
    keras.layers.Dense(1)   # Output layer    
])

# Compile the model
model.compile(optimizer='adam',loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train,Y_train, epochs=100, verbose=1)

# Make predictions
predictions = model.predict(X_test)
print("Predicted prices",predictions.flatten())


# Explain the models predictions
explainer = shap.Explainer(model,X_train)
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test, feature_names=['Area', 'Bedrooms', 'Location'])

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['Area','Bedrooms','Location'], class_names=['Price'], mode='regression')


# Explain Single prediction
i = 0 # Pick a test sample
exp = explainer.explain_instance(X_test[i], model.predict)
exp.show_in_notebook()