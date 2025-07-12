import joblib
from tensorflow.keras.models import load_model

def build_hybrid_model(input_shape=(128, 128, 1), ppg_len=500):
    """
    Same CNN + BiLSTM hybrid model as before
    """
    from tensorflow.keras import layers, models, Input

    # --- Image input branch ---
    img_input = Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(img_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Reshape((-1, x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # --- PPG input branch ---
    ppg_input = Input(shape=(ppg_len, 1), name="ppg_input")
    y = layers.Conv1D(32, 5, activation="relu", padding="same")(ppg_input)
    y = layers.MaxPooling1D(2)(y)
    y = layers.Bidirectional(layers.LSTM(64))(y)
    y = layers.Dense(64, activation="relu")(y)

    # --- Merge ---
    merged = layers.concatenate([x, y])
    z = layers.Dense(64, activation="relu")(merged)
    output = layers.Dense(2, activation="linear", name="bp_output")(z)

    model = models.Model(inputs=[img_input, ppg_input], outputs=output)
    return model

def load_bp_model_and_scaler(model_path, scaler_path):
    """
    Loads trained Keras model and label scaler.
    """
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler
