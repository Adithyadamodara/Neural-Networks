from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

model.compile(
    optimizer='adam',
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)
model.save("property_price_model_clean.h5")