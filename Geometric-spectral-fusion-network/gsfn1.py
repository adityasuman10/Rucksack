import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, Sequential, utils
from keras.optimizers import Adam, AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import cv2
from pathlib import Path
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class EnhancedGeometricSpectralFusionNetwork:
    def __init__(self, input_shape=(224, 224, 1), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def squeeze_excitation_block(self, x, filters, ratio=16):
        """Squeeze-and-Excitation block for channel attention"""
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters // ratio, activation='relu', kernel_regularizer=l2(1e-4))(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, filters))(se)
        return layers.Multiply()([x, se])
    
    def cbam_attention(self, x, filters):
        """Convolutional Block Attention Module (CBAM) - Keras 3 compatible"""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
        max_pool = layers.GlobalMaxPooling2D(keepdims=True)(x)
        
        # Shared MLP
        dense1 = layers.Dense(filters // 8, activation='relu', kernel_regularizer=l2(1e-4))
        dense2 = layers.Dense(filters, kernel_regularizer=l2(1e-4))
        
        avg_pool = dense2(dense1(avg_pool))
        max_pool = dense2(dense1(max_pool))
        
        channel_attention = layers.Activation('sigmoid')(layers.Add()([avg_pool, max_pool]))
        x = layers.Multiply()([x, channel_attention])
        
        # Spatial attention - Keras 3 compatible
        avg_pool_spatial = layers.Lambda(lambda x: keras.ops.mean(x, axis=-1, keepdims=True))(x)
        max_pool_spatial = layers.Lambda(lambda x: keras.ops.max(x, axis=-1, keepdims=True))(x)
        spatial_concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
        
        return layers.Multiply()([x, spatial_attention])
    
    def residual_block(self, x, filters, kernel_size=3, stride=1):
        """Enhanced residual block with attention"""
        shortcut = x
        
        # First conv block
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                         kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second conv block
        x = layers.Conv2D(filters, kernel_size, padding='same', 
                         kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                   kernel_regularizer=l2(1e-4))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut and apply attention
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        x = self.squeeze_excitation_block(x, filters)
        
        return x
    
    def geometric_feature_extractor(self, inputs):
        """Enhanced geometric feature extractor with consistent downsampling"""
        # Initial feature extraction - 224x224 -> 112x112
        x = layers.Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)  # 112x112 -> 56x56
        
        # Multi-scale residual blocks
        # Stage 1 - maintain 56x56
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)
        
        # Stage 2 - 56x56 -> 28x28
        x = self.residual_block(x, 128, stride=2)
        x = self.residual_block(x, 128)
        
        # Multi-scale feature extraction at 28x28
        scale1 = layers.Conv2D(128, 1, padding='same', kernel_regularizer=l2(1e-4))(x)
        scale1 = layers.BatchNormalization()(scale1)
        scale1 = layers.ReLU()(scale1)
        
        scale2 = layers.Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
        scale2 = layers.BatchNormalization()(scale2)
        scale2 = layers.ReLU()(scale2)
        
        scale3 = layers.Conv2D(128, 5, padding='same', kernel_regularizer=l2(1e-4))(x)
        scale3 = layers.BatchNormalization()(scale3)
        scale3 = layers.ReLU()(scale3)
        
        # Combine scales
        geometric_features = layers.Concatenate()([scale1, scale2, scale3])
        geometric_features = self.cbam_attention(geometric_features, 384)
        
        return geometric_features  # Shape: (None, 28, 28, 384)
    
    def spectral_feature_extractor(self, inputs):
        """Enhanced spectral feature extractor with matching spatial dimensions"""
        # Initial processing - 224x224 -> 112x112
        x = layers.Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)  # 112x112 -> 56x56
        
        # Additional downsampling to match geometric features - 56x56 -> 28x28
        x = layers.Conv2D(128, 3, strides=2, padding='same', kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Multi-frequency analysis with different dilation rates at 28x28
        high_freq = layers.SeparableConv2D(
            128, 3, padding='same', dilation_rate=1,
            depthwise_regularizer=l2(1e-4),
            pointwise_regularizer=l2(1e-4)
        )(x)
        high_freq = layers.BatchNormalization()(high_freq)
        high_freq = layers.ReLU()(high_freq)
        
        mid_freq = layers.SeparableConv2D(
            128, 3, padding='same', dilation_rate=2,
            depthwise_regularizer=l2(1e-4),
            pointwise_regularizer=l2(1e-4)
        )(x)
        mid_freq = layers.BatchNormalization()(mid_freq)
        mid_freq = layers.ReLU()(mid_freq)
        
        low_freq = layers.SeparableConv2D(
            128, 3, padding='same', dilation_rate=4,
            depthwise_regularizer=l2(1e-4),
            pointwise_regularizer=l2(1e-4)
        )(x)
        low_freq = layers.BatchNormalization()(low_freq)
        low_freq = layers.ReLU()(low_freq)
        
        # Combine spectral components
        spectral_features = layers.Concatenate()([high_freq, mid_freq, low_freq])
        spectral_features = self.cbam_attention(spectral_features, 384)
        
        return spectral_features  # Shape: (None, 28, 28, 384)
    
    def cross_attention_fusion(self, geo_features, spec_features):
        """Enhanced cross-attention fusion module - Keras 3 compatible"""
        # Feature alignment
        geo_aligned = layers.Conv2D(256, 1, kernel_regularizer=l2(1e-4))(geo_features)
        spec_aligned = layers.Conv2D(256, 1, kernel_regularizer=l2(1e-4))(spec_features)
        
        # Custom Multi-head attention layer for Keras 3
        class CustomMultiHeadAttention(layers.Layer):
            def __init__(self, num_heads=8, key_dim=32, **kwargs):
                super().__init__(**kwargs)
                self.num_heads = num_heads
                self.key_dim = key_dim
                self.attention = layers.MultiHeadAttention(
                    num_heads=num_heads, 
                    key_dim=key_dim,
                    kernel_regularizer=l2(1e-4)
                )
                
            def call(self, query, key, value):
                # Get shape information
                batch_size = keras.ops.shape(query)[0]
                height = keras.ops.shape(query)[1]
                width = keras.ops.shape(query)[2]
                channels = keras.ops.shape(query)[3]
                
                # Reshape for attention computation
                query_flat = keras.ops.reshape(query, (batch_size, -1, channels))
                key_flat = keras.ops.reshape(key, (batch_size, -1, channels))
                value_flat = keras.ops.reshape(value, (batch_size, -1, channels))
                
                # Apply attention
                attention_output = self.attention(query_flat, key_flat, value_flat)
                
                # Reshape back to spatial dimensions
                attention_output = keras.ops.reshape(attention_output, (batch_size, height, width, channels))
                return attention_output
        
        # Initialize attention layers
        attention_layer = CustomMultiHeadAttention(num_heads=8, key_dim=32)
        
        # Cross attention: geo -> spec and spec -> geo
        geo_to_spec = attention_layer(spec_aligned, geo_aligned, geo_aligned)
        spec_to_geo = attention_layer(geo_aligned, spec_aligned, spec_aligned)
        
        # Residual connections
        geo_enhanced = layers.Add()([geo_aligned, spec_to_geo])
        spec_enhanced = layers.Add()([spec_aligned, geo_to_spec])
        
        # Final fusion - now both tensors have shape (None, 28, 28, 256)
        fused = layers.Concatenate()([geo_enhanced, spec_enhanced])
        fused = layers.Conv2D(512, 3, padding='same', kernel_regularizer=l2(1e-4))(fused)
        fused = layers.BatchNormalization()(fused)
        fused = layers.ReLU()(fused)
        fused = layers.Dropout(0.2)(fused)
        
        return fused
    
    def build_model(self):
        """Build the enhanced network"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layer (built into model)
        x = layers.RandomRotation(0.1)(inputs)
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomFlip("horizontal")(x)
        
        # Extract features
        geometric_features = self.geometric_feature_extractor(x)
        spectral_features = self.spectral_feature_extractor(x)
        
        # Print shapes for debugging
        print(f"Geometric features shape: {geometric_features.shape}")
        print(f"Spectral features shape: {spectral_features.shape}")
        
        # Fusion
        fused_features = self.cross_attention_fusion(geometric_features, spectral_features)
        
        # Final processing
        x = layers.Conv2D(1024, 3, padding='same', kernel_regularizer=l2(1e-4))(fused_features)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head with progressive dropout
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        
        # Output with label smoothing built into loss
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name="EnhancedGeometricSpectralFusionNetwork")
        return self.model

class EnhancedISARDataLoader:
    def __init__(self, data_path, image_size=(224, 224)):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.class_names = []
        
    def enhanced_preprocessing(self, image):
        """Enhanced image preprocessing"""
        # Histogram equalization for better contrast
        image = cv2.equalizeHist(image.astype(np.uint8))
        
        # Gaussian noise reduction
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Normalize to [-1, 1] for better gradient flow
        image = (image.astype(np.float32) - 127.5) / 127.5
        
        return image
    
    def load_data(self):
        """Enhanced data loading with better preprocessing"""
        images = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        self.class_names = sorted([d.name for d in class_dirs])  # Sort for consistency
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            if class_name not in self.class_names:
                continue
                
            actual_class_idx = self.class_names.index(class_name)
            print(f"Loading class {class_name} (index {actual_class_idx})...")
            
            # Load images from class directory
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(ext)))
                image_files.extend(list(class_dir.glob(ext.upper())))
            
            for img_file in image_files:
                try:
                    # Read image
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize with proper interpolation
                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
                    
                    # Enhanced preprocessing
                    img = self.enhanced_preprocessing(img)
                    
                    # Add channel dimension
                    img = np.expand_dims(img, axis=-1)
                    
                    images.append(img)
                    labels.append(actual_class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
            
            print(f"Loaded {len([l for l in labels if l == actual_class_idx])} images from {class_name}")
        
        return np.array(images), np.array(labels)
    
    def preprocess_data(self, X, y):
        """Enhanced data preprocessing with stratified split"""
        # Convert labels to categorical
        y_categorical = utils.to_categorical(y, num_classes=len(self.class_names))
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, 
            stratify=np.argmax(y_train, axis=1)
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

def create_advanced_data_generator(X, y, batch_size=32, augment=True):
    """Advanced data generator with sophisticated augmentation"""
    def advanced_augment(image):
        """Advanced augmentation pipeline"""
        img = image.squeeze()
        
        # Convert back to [0, 255] for OpenCV operations
        img = ((img + 1) * 127.5).astype(np.uint8)
        
        # Random elastic deformation
        if np.random.random() > 0.7:
            h, w = img.shape
            dx = np.random.uniform(-5, 5, (h//10, w//10)).astype(np.float32)
            dy = np.random.uniform(-5, 5, (h//10, w//10)).astype(np.float32)
            dx = cv2.resize(dx, (w, h))
            dy = cv2.resize(dy, (w, h))
            
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Random perspective transform
        if np.random.random() > 0.8:
            h, w = img.shape
            margin = min(h, w) * 0.1
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[np.random.uniform(0, margin), np.random.uniform(0, margin)],
                              [w - np.random.uniform(0, margin), np.random.uniform(0, margin)],
                              [np.random.uniform(0, margin), h - np.random.uniform(0, margin)],
                              [w - np.random.uniform(0, margin), h - np.random.uniform(0, margin)]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Convert back to [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 127.5
        return np.expand_dims(img, axis=-1)
    
    while True:
        indices = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_indices = indices[start:end]
            
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                img = X[idx]
                if augment:
                    img = advanced_augment(img)
                batch_x.append(img)
                batch_y.append(y[idx])
            
            yield np.array(batch_x), np.array(batch_y)




def main():
    # Enhanced configuration
    DATA_PATH = r"/content/drive/MyDrive/concave"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16 
    EPOCHS = 50
    
    # Load data
    print("Loading ISAR dataset...")
    data_loader = EnhancedISARDataLoader(DATA_PATH, IMAGE_SIZE)
    
    try:
        X, y = data_loader.load_data()
        print(f"Loaded {len(X)} images with {len(data_loader.class_names)} classes")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for i, (cls, count) in enumerate(zip(data_loader.class_names, counts)):
            if i < len(counts):
                print(f"  {cls}: {counts[i]} images")
        
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(X, y)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Build enhanced model
        print("Building Enhanced Geometric-Spectral Fusion Network...")
        gsfn = EnhancedGeometricSpectralFusionNetwork(
            input_shape=(224, 224, 1),
            num_classes=len(data_loader.class_names)
        )
        model = gsfn.build_model()
        
        # Calculate steps per epoch for cosine decay
        steps_per_epoch = len(X_train) // BATCH_SIZE
        
        # Use Keras 3's built-in cosine decay schedule
        initial_learning_rate = 0.001
        cosine_decay = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=EPOCHS * steps_per_epoch,
            alpha=1e-7
        )
        
        # Compile with cosine decay schedule
        model.compile(
            optimizer=AdamW(learning_rate=cosine_decay, weight_decay=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        model.summary()
        # Print model summary
        print(f"Model has {model.count_params():,} parameters")
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_enhanced_gsfn_model.keras', monitor='val_accuracy', 
                          save_best_only=True, verbose=1)
        ]
        
        # Create advanced data generators
        train_generator = create_advanced_data_generator(X_train, y_train, BATCH_SIZE, augment=True)
        val_generator = create_advanced_data_generator(X_val, y_val, BATCH_SIZE, augment=False)
        
        validation_steps = len(X_val) // BATCH_SIZE
        
        # Train model
        print("Training enhanced model...")
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_generator = create_advanced_data_generator(X_test, y_test, BATCH_SIZE, augment=False)
        test_steps = len(X_test) // BATCH_SIZE
        
        test_results = model.evaluate(test_generator, steps=test_steps, verbose=1)
        print(f"Test Accuracy: {test_results[1]:.4f}")
        print(f"Test Top-K Accuracy: {test_results[2]:.4f}")
        
        # Detailed predictions
        y_pred = model.predict(test_generator, steps=test_steps)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test[:len(y_pred_classes)], axis=1)
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=data_loader.class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=data_loader.class_names,
                   yticklabels=data_loader.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Enhanced training history plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        # Approximate cosine annealing pattern for visualization
        epochs = range(len(history.history['loss']))
        lr_pattern = [initial_learning_rate * (1 + np.cos(np.pi * e / EPOCHS)) / 2 for e in epochs]
        plt.plot(lr_pattern, linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save final model
        model.save('enhanced_geometric_spectral_fusion_network.keras')
        print("Enhanced model saved successfully!")
        
        # Print final performance summary
        print("\n" + "="*50)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"Test Top-K Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
        print(f"Model Parameters: {model.count_params():,}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()