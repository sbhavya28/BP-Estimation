import numpy as np

def predict_bp(model, scaler, scalogram_image, ppg_segment):
    """
    Predict SBP and DBP from a single scalogram image and PPG segment.
    Returns denormalized (real) SBP and DBP in mmHg.
    """
    # Model expects inputs: scalogram and PPG segment
    scaled_img = scalogram_image / 255.0
    scaled_img = scaled_img.reshape(1, 128, 128, 1)

    scaled_ppg = ppg_segment.reshape(1, -1, 1)  # assuming already preprocessed

    # Predict (normalized space)
    pred_norm = model.predict([scaled_img, scaled_ppg])
    
    # Inverse-transform using saved scaler to get mmHg
    pred_mmhg = scaler.inverse_transform(pred_norm)

    sbp, dbp = pred_mmhg[0]
    return sbp, dbp
