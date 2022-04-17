import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def build_model():
    """
    Prepare the model.

    Returns
    -------
    model : model class from any toolbox you choose to use.
        Model definition (untrained).
    """

    model = Sequential([
        # layers.Conv2D(32, 5, padding='same', activation='relu', kernel_regularizer='l2'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(64, 5, padding='same', activation='relu', kernel_regularizer='l2'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer='l2'),
        # layers.MaxPooling2D(),
        # layers.Flatten(),
        layers.Dense(7, activation='relu', kernel_regularizer='l2'),
        layers.Dense(256, activation='relu', kernel_regularizer='l2'),
        layers.Dense(256, activation='relu', kernel_regularizer='l2'),
        layers.Dense(64, activation='relu', kernel_regularizer='l2'),
        layers.Dense(16, activation='relu', kernel_regularizer='l2'),
        layers.Dense(4, activation='softmax', kernel_regularizer='l2')
    ])

    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    return model


def train_model(model, n_validation, features, model_path, epochs=15, write_to_file=False):
    """
    Fit the model on the training data set.

    Arguments
    ---------
    epochs: int
        Number of epochs to train the model for
    model : model class
        Model structure to fit, as defined by build_model().
    n_validation : int
        Number of training examples used for cross-validation.
    write_to_file : bool
        Write model to file; can later be loaded through load_model().

    Returns
    -------
    model : model class
        The trained model.
    """

    training_features, training_labels = features

    model.fit(
        training_features, training_labels,
        #validation_data=(validation_features, validation_labels),
        epochs=epochs)

    if write_to_file:
        model.save(model_path)

    return model

def load_model(Trained_model):
    """
    Load a model from file.

    Returns
    -------
    model : model class
        Previously trained model.
    """
    model = keras.models.load_model(Trained_model) # Load model state

    return model

def evaluate_model(model: Sequential, features):
    """
    Evaluate model on the test set.

    Arguments
    ---------
    model : model class
        Trained model.

    Returns
    -------
    score : float
        A measure of model performance.
    """
    if type(model) != Sequential:
        raise TypeError

    test_features, test_labels = features  # Load test images

    # Evaluate the model on the test data
    print("Evaluate on test data")
    results = model.evaluate(test_features, test_labels, batch_size=128)
    print(f"Test loss: {results[0]:.3f} Test accuracy:{results[1] * 100:.2f}%")