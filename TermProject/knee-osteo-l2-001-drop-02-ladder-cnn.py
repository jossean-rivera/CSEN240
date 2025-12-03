#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Modeling Knee Osteoarthritis [PyTorch Version]')
print('=============================================')
print('Current test:')
print('- Using ResNet-50 due to torchvision compatibility')
print('- L2 Regularization (weight_decay) with penalty 1e-3')
print('- Set learning rate to 9e-5')
print('- **New Classifier Head: 512 -> 1024 -> 512 -> 256 -> 128**')
print('- Enhanced dropout strategy')
print('=============================================')

# Set learning rate
LR = 9e-5
# --- PyTorch Change: Set epochs and patience ---
EPOCHS = 250
PATIENCE = 5


# In[2]:


import numpy as np
import pandas as pd
import os
import time
import shutil
import pathlib
import itertools
from PIL import Image # --- PyTorch Change: Using PIL for image loading ---
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[3]:


# --- This section is identical to your original script ---
data_path = "./"
categories = ["Normal", "Osteopenia", "Osteoporosis"]

image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(data_path, "train", category)
    # Check if directory exists
    if not os.path.isdir(category_path):
        print(f"Warning: Directory not found, skipping: {category_path}")
        continue
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image_paths.append(image_path)
        labels.append(category)

if not image_paths:
    print("\n" + "="*80)
    print("FATAL ERROR: No images found. Check your 'data_path' and directory structure.")
    print(f"Expected structure: {data_path}/train/Normal/image.jpg, etc.")
    print("="*80)
    # exit() # Uncomment this to stop the script if no data is found

# In[4]:


# --- This section is identical to your original script ---
df = pd.DataFrame({"image_path": image_paths, "label": labels})
if df.empty:
    print("DataFrame is empty. Cannot proceed.")
else:
    print('Data Frame dimentions:', df.shape)


# In[5]:


# --- This section is identical to your original script ---
if not df.empty:
    print('Dupplicated sum:', df.duplicated().sum())
    print('IsNull sum:')
    print(df.isnull().sum())
    print('Data Frame Info')
    df.info()

    print("Unique labels: {}".format(df['label'].unique()))
    print("Label counts: {}".format(df['label'].value_counts()))


# In[6]:


# --- This section is identical to your original script (visualization) ---
if not df.empty:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x="label", palette="viridis", ax=ax)
    ax.set_title("Distribution of Tumor Types", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tumor Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, color='black',
            xytext=(0, 5), textcoords='offset points')
    #plt.show() # Disabled for non-interactive script

    label_counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(label_counts))
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
    startangle=140, colors=colors, textprops={'fontsize': 12, 'weight':
    'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax.set_title("Distribution of Tumor Types - Pie Chart", fontsize=14,
    fontweight='bold')
    #plt.show() # Disabled for non-interactive script


# In[7]:


# --- This section is identical to your original script (visualization) ---
if not df.empty:
    num_images = 5
    plt.figure(figsize=(15, 12))
    for i, category in enumerate(categories):
        category_images = df[df['label'] == category]['image_path'].iloc[:num_images]
        for j, img_path in enumerate(category_images):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(category)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


# In[8]:


# --- This section is identical to your original script ---
if not df.empty:
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])
    # We keep the original 'label' column for reference if needed
    df_processed = df[['image_path', 'category_encoded']]


# In[9]:


# --- This section is identical to your original script ---
if 'df_processed' in locals():
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(df_processed[['image_path']],
    df_processed['category_encoded'])
    df_resampled = pd.DataFrame(X_resampled, columns=['image_path'])
    df_resampled['category_encoded'] = y_resampled
    print("\nClass distribution after oversampling:")
    print(df_resampled['category_encoded'].value_counts())
    print(df_resampled.head())
else:
    print("Skipping resampling as initial DataFrame was empty.")
    df_resampled = pd.DataFrame(columns=['image_path', 'category_encoded'])


# In[10]:


# --- This line `astype(str)` is REMOVED. ---
# PyTorch CrossEntropyLoss prefers integer (long) targets, not strings.
# df_resampled['category_encoded'] = df_resampled['category_encoded'].astype(str)


# In[11]:


# --- This section is identical to your original script ---
if not df_resampled.empty:
    train_df_new, temp_df_new = train_test_split(
    df_resampled,
    train_size=0.8,
    shuffle=True,
    random_state=42,
    stratify=df_resampled['category_encoded']
    )
    print(f"Train shape: {train_df_new.shape}")

    valid_df_new, test_df_new = train_test_split(
    temp_df_new,
    test_size=0.5,
    shuffle=True,
    random_state=42,
    stratify=temp_df_new['category_encoded']
    )
    print(f"Validation shape: {valid_df_new.shape}")
    print(f"Test shape: {test_df_new.shape}")
else:
    print("Skipping data split as resampled DataFrame is empty.")
    # Create empty dataframes to avoid errors later
    train_df_new = pd.DataFrame(columns=['image_path', 'category_encoded'])
    valid_df_new = pd.DataFrame(columns=['image_path', 'category_encoded'])
    test_df_new = pd.DataFrame(columns=['image_path', 'category_encoded'])


# In[12]:


# --- PyTorch Change: Replaced Keras ImageDataGenerator with PyTorch Dataset ---
batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
num_classes = len(categories)

# Define transforms
# Using standard ImageNet mean and std. ToTensor() scales images to [0, 1]
# Keras's `rescale=1./255` is equivalent to just ToTensor()
# Normalization is standard for transfer learning.
data_transforms = T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OsteoarthritisDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['category_encoded']

        # Load image using PIL
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# Create Datasets
train_dataset = OsteoarthritisDataset(train_df_new, transform=data_transforms)
valid_dataset = OsteoarthritisDataset(valid_df_new, transform=data_transforms)
test_dataset = OsteoarthritisDataset(test_df_new, transform=data_transforms)

# Create DataLoaders (num_workers=0 fix for SLURM)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


print(f"DataLoaders created.")
print(f"Num GPUs Available: {torch.cuda.device_count()}")


# In[13]:


# --- PyTorch Change: GaussianNoise layer ---
# Keras GaussianNoise layer equivalent in PyTorch
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        stddev (float): A positive float. Standard deviation of the noise.
    """
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

# --- PyTorch Change: Replaced Keras functional model with PyTorch nn.Module ---

class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.2):
        super().__init__()

        # --- Load pre-trained ResNet-50 ---
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Get all layers except the final avgpool and fc
        self.features = nn.Sequential(*list(self.base_model.children())[:-2])

        # Freeze the base model
        for param in self.features.parameters():
            param.requires_grad = False

        feature_channels = 2048 # ResNet-50 output channel depth

        # Replicate your Keras head
        self.dropout_base = nn.Dropout(dropout_rate) # Dropout(0.2)

        # MultiHeadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=8,
            batch_first=True
        )

        self.dropout_attn = nn.Dropout(dropout_rate) # Dropout(0.2)

        self.gaussian_noise = GaussianNoise(0.25) # GaussianNoise(0.25)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # GlobalAveragePooling2D

        # ====================================================================
        # === START OF CHANGES FOR NEW LAYER STRUCTURE: 512->1024->512->256->128 ===
        # ====================================================================

        self.classifier = nn.Sequential(
            # 1. Input: 2048 -> 512
            nn.Linear(feature_channels, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            # 2. **NEW LAYER**: 512 -> 1024
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),

            # 3. **NEW LAYER**: 1024 -> 512
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            # 4. 512 -> 256 (Adjusted input from the new 512 layer)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            # 5. 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            # 6. 128 -> 3 (Output)
            nn.Dropout(0.2), # Dropout_Before_Output

            nn.Linear(128, num_classes) # Output layer
        )

        # ====================================================================
        # === END OF CHANGES FOR NEW LAYER STRUCTURE ===
        # ====================================================================


    def forward(self, x):
        # 1. Base model features
        x = self.features(x) # (N, 2048, 7, 7)

        # 2. Dropout after base
        x = self.dropout_base(x)

        # 3. Prepare for MHA
        n, c, h, w = x.shape
        # (N, 2048, 49) -> (N, 49, 2048)
        x_seq = x.view(n, c, h * w).permute(0, 2, 1)

        # 4. MultiHeadAttention
        attn_output, _ = self.mha(x_seq, x_seq, x_seq) # (N, 49, 2048)

        # 5. Reshape back to spatial
        # (N, 2048, 49) -> (N, 2048, 7, 7)
        x = attn_output.permute(0, 2, 1).view(n, c, h, w)

        # 6. Dropout after attention
        x = self.dropout_attn(x)

        # 7. Gaussian Noise
        x = self.gaussian_noise(x)

        # 8. Global Pooling
        x = self.global_pool(x) # (N, 2048, 1, 1)

        # 9. Flatten
        x = torch.flatten(x, 1) # (N, 2048)

        # 10. Classifier
        x = self.classifier(x) # (N, num_classes)

        return x

# --- Instantiate model, loss, and optimizer ---

cnn_model = ResNetWithAttention(num_classes=num_classes).to(device)


# L2 regularization is 'weight_decay' in the PyTorch optimizer
# Your L2 penalty was 1e-3
optimizer = optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

# Print model summary
print("\n" + "="*80)
print("MODEL ARCHITECTURE (using ResNet-50 base, updated layers)")
print("========================================================")
print(cnn_model)
# A more detailed summary can be obtained with torchsummary
# from torchsummary import summary
# summary(cnn_model, input_size=(3, 224, 224))


# In[14]:


# --- PyTorch Change: Manual Training and Validation Loop ---
# This replaces `model.fit()`

history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

best_val_loss = float('inf')
patience_counter = 0
best_model_weights = None

if train_df_new.empty:
    print("Training DataFrame is empty. Skipping training loop.")
else:
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        cnn_model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # This is the loop that was previously hanging
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct_train / total_train
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        # --- Validation Phase ---
        cnn_model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = cnn_model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(valid_dataset)
        epoch_val_acc = 100 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f'Epoch {epoch+1}/{EPOCHS} | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Train Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {epoch_val_acc:.2f}%')

        # --- Early Stopping Logic ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model weights
            best_model_weights = cnn_model.state_dict()
            torch.save(best_model_weights, 'best_model.pth')
            print(f'   -> New best model saved with val_loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch+1} epochs.')
                break

    print("\n" + "="*80)
    print("TRAINING FINISHED")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)

    # Load the best model weights back
    if best_model_weights:
        cnn_model.load_state_dict(best_model_weights)
        print("Best model weights restored.")


# In[15]:


# --- PPO Loss calculation is commented out ---
# This loss is from reinforcement learning (Proximal Policy Optimization)
# and is not standard for this classification task. It wasn't used
# in your Keras training loop, so I am omitting it.

# y_pred = cnn_model.predict(valid_gen_new) ...
# ppo_loss_value = ppo_loss(y_true, y_pred)
# print("\nPPO Loss on Validation Data:", ppo_loss_value.numpy())


# In[16]:


# --- Plotting -- This section is identical, just uses the 'history' dict ---
if history['accuracy']:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()
else:
    print("No training history to plot.")


# In[17]:


# --- Evaluation on Test Set ---
if test_df_new.empty:
    print("Test DataFrame is empty. Skipping final evaluation.")
else:
    print("\n" + "="*80)
    print("STARTING FINAL EVALUATION ON TEST SET")
    print("="*80)

    cnn_model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = cnn_model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # --- Metrics calculations are identical (using sklearn) ---
    test_labels = np.array(all_labels)
    predicted_classes = np.array(all_predictions)

    # Get class names from the LabelEncoder
    class_names = list(label_encoder.inverse_transform(range(num_classes)))

    # Displays classification report
    report = classification_report(test_labels, predicted_classes,
    target_names=class_names)
    print("Classification Report:")
    print(report)

    # Display confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

print("\nScript finished.")