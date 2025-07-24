import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, Sequential, utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class GeometricSpectralFusionNetwork:
    def __init__(self, input_shape=(224, 224, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def spectral_attention_block(self, x, filters):
        """Spectral attention mechanism for frequency domain features"""
        # Global average pooling for channel attention
        gap = layers.GlobalAveragePooling2D()(x)
        gap = layers.Reshape((1, 1, filters))(gap)
        
        # Channel attention
        channel_attention = layers.Dense(filters // 4, activation='relu')(gap)
        channel_attention = layers.Dense(filters, activation='sigmoid')(channel_attention)
        
        # Apply channel attention
        x_attended = layers.Multiply()([x, channel_attention])
        
        # Spatial attention using FFT-inspired convolutions
        spatial_att = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(x_attended)
        x_final = layers.Multiply()([x_attended, spatial_att])
        
        return x_final
    
    def geometric_feature_extractor(self, inputs):
        """Extract geometric features focusing on shape and structure"""
        # Initial convolution
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Multi-scale geometric feature extraction
        # Scale 1: Fine details
        scale1 = layers.Conv2D(64, 3, padding='same')(x)
        scale1 = layers.BatchNormalization()(scale1)
        scale1 = layers.ReLU()(scale1)
        
        # Scale 2: Medium details
        scale2 = layers.Conv2D(64, 5, padding='same')(x)
        scale2 = layers.BatchNormalization()(scale2)
        scale2 = layers.ReLU()(scale2)
        
        # Scale 3: Coarse details
        scale3 = layers.Conv2D(64, 7, padding='same')(x)
        scale3 = layers.BatchNormalization()(scale3)
        scale3 = layers.ReLU()(scale3)
        
        # Combine multi-scale features
        geometric_features = layers.Concatenate()([scale1, scale2, scale3])
        
        # Geometric attention
        geometric_features = self.spectral_attention_block(geometric_features, 192)
        
        return geometric_features
    
    def spectral_feature_extractor(self, inputs):
        """Extract spectral features focusing on frequency characteristics"""
        # Depthwise separable convolutions for spectral analysis
        x = layers.DepthwiseConv2D(3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, 1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Spectral decomposition inspired blocks
        # High frequency components
        high_freq = layers.Conv2D(64, 3, strides=1, padding='same')(x)
        high_freq = layers.BatchNormalization()(high_freq)
        high_freq = layers.ReLU()(high_freq)
        
        # Mid frequency components
        mid_freq = layers.Conv2D(64, 3, strides=1, padding='same', dilation_rate=2)(x)
        mid_freq = layers.BatchNormalization()(mid_freq)
        mid_freq = layers.ReLU()(mid_freq)
        
        # Low frequency components
        low_freq = layers.Conv2D(64, 3, strides=1, padding='same', dilation_rate=4)(x)
        low_freq = layers.BatchNormalization()(low_freq)
        low_freq = layers.ReLU()(low_freq)
        
        # Combine spectral components
        spectral_features = layers.Concatenate()([high_freq, mid_freq, low_freq])
        
        # Spectral attention
        spectral_features = self.spectral_attention_block(spectral_features, 192)
        
        return spectral_features
    
    def fusion_module(self, geometric_features, spectral_features):
        """Fuse geometric and spectral features"""
        # Feature alignment
        geo_aligned = layers.Conv2D(128, 1)(geometric_features)
        spec_aligned = layers.Conv2D(128, 1)(spectral_features)
        
        # Cross-attention between geometric and spectral features
        # Geometric to Spectral attention
        geo_to_spec = layers.GlobalAveragePooling2D()(geo_aligned)
        geo_to_spec = layers.Dense(128, activation='relu')(geo_to_spec)
        geo_to_spec = layers.Dense(128, activation='sigmoid')(geo_to_spec)
        geo_to_spec = layers.Reshape((1, 1, 128))(geo_to_spec)
        spec_enhanced = layers.Multiply()([spec_aligned, geo_to_spec])
        
        # Spectral to Geometric attention
        spec_to_geo = layers.GlobalAveragePooling2D()(spec_aligned)
        spec_to_geo = layers.Dense(128, activation='relu')(spec_to_geo)
        spec_to_geo = layers.Dense(128, activation='sigmoid')(spec_to_geo)
        spec_to_geo = layers.Reshape((1, 1, 128))(spec_to_geo)
        geo_enhanced = layers.Multiply()([geo_aligned, spec_to_geo])
        
        # Concatenate enhanced features
        fused_features = layers.Concatenate()([geo_enhanced, spec_enhanced])
        
        # Additional fusion processing
        fused = layers.Conv2D(256, 3, padding='same')(fused_features)
        fused = layers.BatchNormalization()(fused)
        fused = layers.ReLU()(fused)
        
        return fused
    
    def build_model(self):
        """Build the complete Geometric-Spectral Fusion Network"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Extract geometric and spectral features
        geometric_features = self.geometric_feature_extractor(inputs)
        spectral_features = self.spectral_feature_extractor(inputs)
        
        # Downsample features
        geometric_features = layers.MaxPooling2D(2)(geometric_features)
        spectral_features = layers.MaxPooling2D(2)(spectral_features)
        
        # Fuse features
        fused_features = self.fusion_module(geometric_features, spectral_features)
        
        # Further processing
        x = layers.MaxPooling2D(2)(fused_features)
        x = layers.Conv2D(512, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(512, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.4)(x)
        
        # Global features
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name="GeometricSpectralFusionNetwork")
        return self.model

class ISARDataLoader:
    def __init__(self, data_path, image_size=(224, 224)):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.class_names = []
        
    def load_data(self):
        """Load ISAR dataset from directory structure"""
        images = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        self.class_names = [d.name for d in class_dirs]
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        for class_idx, class_dir in enumerate(class_dirs):
            print(f"Loading class {class_dir.name}...")
            
            # Load images from class directory
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.bmp')) + \
                         list(class_dir.glob('*.tif')) + list(class_dir.glob('*.tiff'))
            
            for img_file in image_files:
                try:
                    # Read image
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    # Resize image
                    img = cv2.resize(img, self.image_size)
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    # Add channel dimension
                    img = np.expand_dims(img, axis=-1)
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
            
            print(f"Loaded {len([l for l in labels if l == class_idx])} images from {class_dir.name}")
        
        return np.array(images), np.array(labels)
    
    def preprocess_data(self, X, y):
        """Preprocess the data"""
        # Convert labels to categorical
        y_categorical = utils.to_categorical(y, num_classes=len(self.class_names))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# Custom data augmentation function
def augment_data(X_train, y_train, batch_size=32):
    """Custom data augmentation generator"""
    def augment_image(image):
        """Apply random augmentations to an image"""
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random shift
        if np.random.random() > 0.5:
            h, w = image.shape[:2]
            shift_x = int(np.random.uniform(-0.1, 0.1) * w)
            shift_y = int(np.random.uniform(-0.1, 0.1) * h)
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random zoom
        if np.random.random() > 0.5:
            zoom = np.random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, 0, zoom)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)  # Vertical flip
        
        return image
    
    num_samples = len(X_train)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                img = X_train[idx].squeeze()  # Remove channel dimension for augmentation
                augmented_img = augment_image(img)
                augmented_img = np.expand_dims(augmented_img, axis=-1)  # Add channel dimension back
                
                batch_x.append(augmented_img)
                batch_y.append(y_train[idx])
            
            yield np.array(batch_x), np.array(batch_y)

def main():
    # Configuration
    DATA_PATH = r"C:\vscode\concave"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Load data
    print("Loading ISAR dataset...")
    data_loader = ISARDataLoader(DATA_PATH, IMAGE_SIZE)
    
    try:
        X, y = data_loader.load_data()
        print(f"Loaded {len(X)} images with {len(data_loader.class_names)} classes")
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(X, y)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Build model
        print("Building Geometric-Spectral Fusion Network...")
        gsfn = GeometricSpectralFusionNetwork(
            input_shape=(224, 224, 1),
            num_classes=len(data_loader.class_names)
        )
        model = gsfn.build_model()
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']  # Fixed metric name
        )
        
        # Print model summary
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
            ModelCheckpoint('best_gsfn_model.keras', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Create data generator
        train_generator = augment_data(X_train, y_train, BATCH_SIZE)
        steps_per_epoch = len(X_train) // BATCH_SIZE
        
        # Train model
        print("Training model...")
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        test_top_k_accuracy = test_results[2] if len(test_results) > 2 else None
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        if test_top_k_accuracy:
            print(f"Test Top-K Accuracy: {test_top_k_accuracy:.4f}")
        
        # Predictions and classification report
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=data_loader.class_names))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        # Save final model
        model.save('geometric_spectral_fusion_network.keras')
        print("Model saved as 'geometric_spectral_fusion_network.keras'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check if the dataset path exists and contains subdirectories with images.")

if __name__ == "__main__":
    main()