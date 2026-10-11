
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from services.preprocessing import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Rajasthan_Heatwave_2006_2025.csv"
MODEL_DIR = BASE_DIR / "models"

TARGET = "HEATWAVE"
TRAIN_END_YEAR = 2021
RANDOM_SEED = 42

FEATURES = [
    "MONTH",
    "DAY",
    "WIND_U10",
    "WIND_V10",
    "MSLP",
    "BLH",
    "GEOP",
    "TEMP2M",
    "TMAX",
    "TMIN",
    "DEW2M",
    "CLOUD",
    "RAIN",
    "SRAD",
    "EVAP",
    "SOILT1",
    "SOILM1",
    "LAI",
    "LAT",
    "LON",
]


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("HEATWAVE MONITORING SYSTEM — DNN TRAINING")
    print("=" * 60)

    print("\nLoading dataset...")
    df = load_dataset(DATA_PATH)

    required_columns = FEATURES + [TARGET, "YEAR"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    # Keep only valid binary labels and rows with usable target/year.
    df = df.dropna(subset=[TARGET, "YEAR"]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()

    # Chronological holdout: train on earlier years, test on later years.
    train_df = df[df["YEAR"] <= TRAIN_END_YEAR].copy()
    test_df = df[df["YEAR"] > TRAIN_END_YEAR].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "The chronological split produced an empty training or test set. "
            "Check the YEAR values in the dataset."
        )

    # Sort for reproducible chronological processing.
    train_df = train_df.sort_values(
        ["YEAR", "MONTH", "DAY"]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["YEAR", "MONTH", "DAY"]
    ).reset_index(drop=True)

    X_train = train_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test = test_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "The training data must contain both classes: 0 and 1."
        )

    if len(np.unique(y_test)) < 2:
        print(
            "Warning: the test set contains only one class. "
            "Some evaluation metrics may be less informative."
        )

    print(f"Total usable rows: {len(df):,}")
    print(f"Training rows (YEAR <= {TRAIN_END_YEAR}): {len(X_train):,}")
    print(f"Testing rows (YEAR > {TRAIN_END_YEAR}): {len(X_test):,}")
    print(f"Input features: {len(FEATURES)}")
    print("\nTraining target distribution:")
    print(pd.Series(y_train).value_counts().sort_index().to_string())

    # Fit preprocessing on training data only to avoid data leakage.
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Compensate for the imbalance between heatwave and non-heatwave days.
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    print("\nClass weights:", class_weights)

    # --------------------------------------------------
    # BUILD THE DEEP NEURAL NETWORK
    # --------------------------------------------------

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(FEATURES),)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="heatwave_dnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    # Use the last portion of the chronological training rows for validation.
    split_index = int(len(X_train_scaled) * 0.8)

    if split_index == 0 or split_index == len(X_train_scaled):
        raise ValueError("Not enough training data for a validation split.")

    X_fit = X_train_scaled[:split_index]
    y_fit = y_train[:split_index]
    X_val = X_train_scaled[split_index:]
    y_val = y_train[split_index:]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00001,
        ),
    ]

    print("\nTraining DNN...")
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------
    # EVALUATE ON THE HELD-OUT YEARS
    # --------------------------------------------------

    probabilities = model.predict(X_test_scaled, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Heatwave", "Heatwave"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # --------------------------------------------------
    # SAVE MODEL AND PREPROCESSOR
    # --------------------------------------------------

    model_path = MODEL_DIR / "heatwave_dnn.keras"
    preprocessor_path = MODEL_DIR / "dnn_preprocessor.joblib"

    model.save(model_path)

    joblib.dump(
        {
            "features": FEATURES,
            "preprocessor": preprocessor,
            "train_end_year": TRAIN_END_YEAR,
            "threshold": 0.5,
        },
        preprocessor_path,
    )

    print("\nTraining complete.")
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(
        "\nNote: this model classifies heatwave labels for the same-day "
        "weather inputs. It is not a future-day forecasting model."
    )


if __name__ == "__main__":
    main()
