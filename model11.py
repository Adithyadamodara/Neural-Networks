import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# === Load dataset === #
df = pd.read_csv("clean_melb_data.csv")

# === Drop rows with missing target or critical features === #
df.dropna(subset=['Price', 'Lattitude', 'Longtitude'], inplace=True)

# === Select target === #
y = df['Price']

# === Structured Features === #
features = ['Rooms', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
            'Distance', 'Suburb', 'Type', 'Regionname', 'Lattitude', 'Longtitude']

X = df[features]

# === Separate categorical and numeric features === #
cat_features = ['Suburb', 'Type', 'Regionname']
num_features = list(set(features) - set(cat_features))

# === Preprocessing === #
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_processed = preprocessor.fit_transform(X)

# === Image feature placeholder (all zeros) === #
#image_features = np.zeros((X_processed.shape[0], 2048))
image_features = np.random.rand(1, 2048)

# === Train-test split === #
X_train_s, X_test_s, X_train_i, X_test_i, y_train, y_test = train_test_split(
    X_processed, image_features, y, test_size=0.2, random_state=42
)

# === Model === #
structured_input = Input(shape=(X_train_s.shape[1],), name="structured_input")
x1 = layers.Dense(64, activation='relu')(structured_input)
x1 = layers.Dense(32, activation='relu')(x1)

image_input = Input(shape=(2048,), name="image_input")
x2 = layers.Dense(256, activation='relu')(image_input)
x2 = layers.Dense(64, activation='relu')(x2)

combined = layers.concatenate([x1, x2])
output = layers.Dense(1, activation='linear')(combined)

model = models.Model(inputs=[structured_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Train === #
print("🚀 Training the model (no POI features)...")
model.fit([X_train_s, X_train_i], y_train, validation_split=0.1, epochs=10, batch_size=16)

# === Evaluate and Save === #
loss, mae = model.evaluate([X_test_s, X_test_i], y_test)
print(f"📊 Test MAE: ₹ {mae:,.2f}")

# Save model and preprocessor
model.save("property_price_model_clean.h5")
import joblib
joblib.dump(preprocessor, "preprocessor_clean.pkl")
print("✅ Model and preprocessor saved.")
