from end_to_end_training import train_model


def test_end_to_end_linear():

    NUM_EPOCHS=10

    NUMERIC_FEATURES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
    ]

    train_model(
        numeric_features=NUMERIC_FEATURES,
        num_epochs=NUM_EPOCHS,
    )
