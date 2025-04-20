import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras import layers, models, Input
import joblib

# === Load dataset === #
df = pd.read_csv("clean_melb_data.csv")

# === Drop rows with missing target or critical features === #
df.dropna(subset=['Price', 'Lattitude', 'Longtitude'], inplace=True)

# === Target === #
y = df['Price']

# === Features (no POIs, no image) === #
features = ['Rooms', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
            'Distance', 'Suburb', 'Type', 'Regionname', 'Lattitude', 'Longtitude']

X = df[features]

# === Preprocessing === #
cat_features = ['Suburb', 'Type', 'Regionname']
num_features = list(set(features) - set(cat_features))

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_processed = preprocessor.fit_transform(X)

# === Split === #
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# === Model === #
structured_input = Input(shape=(X_train.shape[1],), name="structured_input")
x = layers.Dense(64, activation='relu')(structured_input)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

model = models.Model(inputs=structured_input, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Train === #
print("🚀 Training structured-only model...")
model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=16)

# === Evaluate === #
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Test MAE: ₹ {mae:,.2f}")

# === Save model and preprocessor === #
model.save("structured_price_model.h5")
joblib.dump(preprocessor, "structured_preprocessor.pkl")
print("📦 Model and preprocessor saved.")
