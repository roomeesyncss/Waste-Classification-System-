

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import sys

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPU is available: {physical_devices}")

    try:
        tf.config.optimizer.set_jit(True)  # Enable XLA acceleration if available
        print("XLA acceleration enabled if available")
    except:
        print("XLA acceleration not available")

    # Apply additional memory optimizations
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth configured")
    except:
        print("Could not apply GPU memory optimizations")
else:
    print("No GPU found, using CPU")

try:
    from tensorflow.keras.mixed_precision import set_global_policy

    set_global_policy('mixed_float16')
    print("Mixed precision training enabled")
except ImportError:
    # For older TensorFlow versions
    try:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled (legacy)")
    except:
        print("Mixed precision training not available")

IMG_SIZE = 160
BATCH_SIZE = 64  #
EPOCHS = 15
LEARNING_RATE = 0.0002
DATA_DIR = 'data'


def create_data_generators():
    print(f"Creating data generators from directory: {DATA_DIR}")

    # Check if the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found!")
        print("Please create a 'data' directory with subdirectories for each class.")
        print("Example structure:")
        print("data/")
        print("  ├── cardboard/")
        print("  ├── glass/")
        print("  ├── metal/")
        print("  ├── paper/")
        print("  ├── plastic/")
        print("  └── trash/")
        sys.exit(1)

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(classes) == 0:
        print("No class directories found in data folder!")
        sys.exit(1)

    print(f"Found classes: {classes}")

    # Count images per class
    total_images = 0
    for cls in classes:
        class_dir = os.path.join(DATA_DIR, cls)
        image_files = [f for f in os.listdir(class_dir)
                       if os.path.isfile(os.path.join(class_dir, f)) and
                       f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        print(f"Class {cls}: {len(image_files)} images")
        total_images += len(image_files)

    print(f"Total images: {total_images}")

    if total_images < 100:
        print("Warning: Very few images found. Consider adding more data for better training.")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalize pixel values
        rotation_range=15,  # R
        width_shift_range=0.1,
        height_shift_range=0.1,  #
        zoom_range=0.1,  #
        horizontal_flip=True,  # Keep flip
        validation_split=0.3  #
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.5  # This splits the validation set in half (15% for validation, 15% for test)
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    validation_generator = test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    print(f"Class indices: {class_indices}")

    return train_generator, validation_generator, test_generator, class_names


def create_model(model_type, num_classes):
    """Create and compile a model based on the specified pre-trained architecture."""
    print(f"Creating {model_type} model...")

    if model_type == 'VGG16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif model_type == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(256, activation='relu')(x)  # REDUCED from 512 to 256
    x = Dropout(0.5)(x)  # Dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer with softmax

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print(f"{model_type} model created with {num_classes} output classes")
    print(f"Total parameters: {model.count_params():,}")

    return model


def create_callbacks(model_name):
    """Create callbacks for model training."""
    # Save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        f'models/{model_name}_best.keras',  # Changed to .keras format
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # OPTIMIZATION: More aggressive early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # REDUCED from 10 to 5
        restore_best_weights=True,
        verbose=1
    )

    # OPTIMIZATION: More aggressive learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Factor by which to reduce learning rate
        patience=3,  # REDUCED from 5 to 3
        min_lr=1e-6,  # Minimum learning rate
        verbose=1
    )

    return [checkpoint, early_stopping, reduce_lr]


def plot_training_history(history, model_name):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png')
    plt.close()

    print(f"Training history plots saved to 'plots/{model_name}_training_history.png'")


def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar()

    # Add axis labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add count numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()

    print(f"Confusion matrix saved to 'plots/{model_name}_confusion_matrix.png'")


def evaluate_model(model, test_generator, class_names, model_name):
    """Evaluate model on test data and generate metrics."""
    print(f"\nEvaluating {model_name} model...")

    # Get number of test batches
    steps = len(test_generator)

    # Get predictions
    y_pred_prob = model.predict(test_generator, steps=steps)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # True labels (test_generator.classes only contains labels for the first batch)
    test_generator.reset()  # Reset generator to start from beginning
    y_true = np.array([])
    for i in range(steps):
        batch_x, batch_y = next(test_generator)
        y_true = np.append(y_true, np.argmax(batch_y, axis=1))
        if len(y_true) >= len(y_pred):
            break

    # Trim predictions to match true labels length
    y_pred = y_pred[:len(y_true)]

    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, model_name)

    # Save metrics to file
    with open(f'results/{model_name}_evaluation.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.datetime.now()}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Print metrics
    print("\nClassification Report:")
    print(report)

    # Extract overall accuracy from the report
    lines = report.split('\n')
    for line in lines:
        if 'accuracy' in line:
            accuracy = float(line.split()[-2])
            return accuracy

    # Fallback: calculate accuracy manually
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def fine_tune_model(model_path, model_type, train_generator, validation_generator,
                    class_names, num_layers_to_unfreeze=10):
    """Fine-tune a trained model by unfreezing some layers."""
    print(f"\nFine-tuning {model_type} model...")

    # Load the best model from training
    model = tf.keras.models.load_model(model_path)

    # FIXED: Get the base model correctly
    base_model = model.layers[0]  # The first layer should be the base model

    # Check if it has layers attribute (it should for VGG16 and MobileNetV2)
    if hasattr(base_model, 'layers'):
        # Determine which layers to unfreeze
        if model_type == 'VGG16':
            # Unfreeze the last few convolutional blocks
            for layer in base_model.layers[-num_layers_to_unfreeze:]:
                layer.trainable = True
                print(f"Unfrozen layer: {layer.name}")
        elif model_type == 'MobileNetV2':
            # Unfreeze the last few blocks
            for layer in base_model.layers[-num_layers_to_unfreeze:]:
                layer.trainable = True
                print(f"Unfrozen layer: {layer.name}")
    else:
        print(f"Warning: Base model doesn't have layers attribute. Skipping fine-tuning.")
        return model_path

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = create_callbacks(f"{model_type}_fine_tuned")

    # Fine-tune the model
    fine_tune_epochs = max(5, EPOCHS // 3)
    print(f"Fine-tuning {model_type} model with {fine_tune_epochs} epochs...")

    try:
        history = model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )

        # Plot training history
        plot_training_history(history, f"{model_type}_fine_tuned")

        # Save the final fine-tuned model
        model.save(f'models/{model_type}_fine_tuned_final.keras')
        print(f"Fine-tuned model saved to 'models/{model_type}_fine_tuned_final.keras'")

        return f'models/{model_type}_fine_tuned_best.keras'

    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return model_path


def train_and_evaluate(model_type, train_generator, validation_generator, test_generator, class_names):
    """Train, evaluate, and fine-tune a model of the specified type."""
    print(f"\n{'=' * 50}")
    print(f"Training {model_type} model")
    print(f"{'=' * 50}")

    # Create model
    model = create_model(model_type, len(class_names))

    # OPTIMIZATION: Adjust training parameters based on model type
    current_epochs = EPOCHS
    if model_type == 'VGG16':
        current_epochs = max(8, EPOCHS - 5)  # Reduce epochs for VGG16
        print(f"Reducing epochs to {current_epochs} for VGG16 to speed up training")

    # Create callbacks
    callbacks = create_callbacks(model_type)

    # Train model
    print(f"Training {model_type} model for {current_epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=current_epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Plot training history
    plot_training_history(history, model_type)

    # Save final model
    model.save(f'models/{model_type}_final.keras2')
    print(f"Final model saved to 'models/{model_type}_2final.keras'")

    # Load the best model for evaluation
    best_model = tf.keras.models.load_model(f'models/{model_type}_best.keras3')

    # Evaluate model
    accuracy = evaluate_model(best_model, test_generator, class_names, model_type)

    # Fine-tune only MobileNetV2 to save time
    if model_type == 'MobileNetV2':
        fine_tuned_model_path = fine_tune_model(
            f'models/{model_type}_best.keras',
            model_type,
            train_generator,
            validation_generator,
            class_names
        )

        # Load and evaluate fine-tuned model if it exists
        if os.path.exists(fine_tuned_model_path):
            try:
                fine_tuned_model = tf.keras.models.load_model(fine_tuned_model_path)
                fine_tuned_accuracy = evaluate_model(
                    fine_tuned_model,
                    test_generator,
                    class_names,
                    f"{model_type}_fine_tuned"
                )
                return max(accuracy, fine_tuned_accuracy)
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")

    return accuracy


def main():
    print("Starting waste classification model training (fixed version)")
    print("TensorFlow version:", tf.__version__)

    try:
        train_generator, validation_generator, test_generator, class_names = create_data_generators()
    except Exception as e:
        print(f"Error creating data generators: {e}")
        return

    print(f"\nDataset summary:")
    print(f"- Classes: {len(class_names)}")
    print(f"- Training samples: {train_generator.samples}")
    print(f"- Validation samples: {validation_generator.samples}")
    print(f"- Test samples: {test_generator.samples}")

    # Train MobileNetV2 first as it's faster
    try:
        mobilenetv2_accuracy = train_and_evaluate(
            'MobileNetV2',
            train_generator,
            validation_generator,
            test_generator,
            class_names
        )
    except Exception as e:
        print(f"Error training MobileNetV2: {e}")
        mobilenetv2_accuracy = 0.0

    # Then train VGG16
    try:
        vgg16_accuracy = train_and_evaluate(
            'VGG16',
            train_generator,
            validation_generator,
            test_generator,
            class_names
        )
    except Exception as e:
        print(f"Error training VGG16: {e}")
        vgg16_accuracy = 0.0

    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(f"VGG16 Accuracy: {vgg16_accuracy:.4f}")
    print(f"MobileNetV2 Accuracy: {mobilenetv2_accuracy:.4f}")

    # Determine best model
    best_model = "VGG16" if vgg16_accuracy > mobilenetv2_accuracy else "MobileNetV2"
    print(f"\nBest model: {best_model}")

    # Save model info
    with open('models/best_model_info.txt', 'w') as f:
        f.write(f"Best model: {best_model}\n")
        f.write(f"VGG16 Accuracy: {vgg16_accuracy:.4f}\n")
        f.write(f"MobileNetV2 Accuracy: {mobilenetv2_accuracy:.4f}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")

    print("\nTraining and evaluation complete")
    print(f"See 'models/best_model_info.txt' for best model information")


if __name__ == "__main__":
    main()