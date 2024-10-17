import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE

# Load data
y_tr = pd.read_csv('tox21_labels_train.csv.gz', index_col=0, compression="gzip")
y_te = pd.read_csv('tox21_labels_test.csv.gz', index_col=0, compression="gzip")
x_tr_dense = pd.read_csv('tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
x_te_dense = pd.read_csv('tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
x_tr_sparse = io.mmread('tox21_sparse_train.mtx.gz').tocsc()
x_te_sparse = io.mmread('tox21_sparse_test.mtx.gz').tocsc()

# Filter and concatenate dense + sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

# Standardize features with tanh scaling
scaler = StandardScaler()
x_tr = np.tanh(scaler.fit_transform(x_tr))
x_te = np.tanh(scaler.transform(x_te))

# Learning rate decay function
def lr_schedule(epoch, lr):
    return lr * 0.95 if epoch > 10 else lr

# Create a shared multi-task model architecture with LeakyReLU
def create_model(input_dim):
    model = Sequential([
        Dense(1024, input_dim=input_dim, kernel_regularizer=l2(1e-5)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Dense(512, kernel_regularizer=l2(1e-5)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        Dense(1, activation='sigmoid')  # Single output for binary classification
    ])
    return model

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train and evaluate the model for each task independently
auc_scores = []

for i, target in enumerate(y_tr.columns):
    print(f"\nTraining on assay: {target}")

    # Select valid rows for the current task
    valid_rows = np.isfinite(y_tr[target]).values
    x_target, y_target = x_tr[valid_rows], y_tr[target][valid_rows]

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    x_balanced, y_balanced = smote.fit_resample(x_target, y_target)

    # Split into train/validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # Create and compile a new model and optimizer for the current task
    model = create_model(input_dim=x_tr.shape[1])
    optimizer = Adam(learning_rate=1e-3)  # Create a new optimizer instance
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

    # Train the model
    model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=100, batch_size=256, callbacks=[early_stopping, reduce_lr], verbose=2
    )

    # Evaluate on the test set
    valid_test_rows = np.isfinite(y_te[target]).values
    y_test = y_te[target][valid_test_rows].values
    p_test = model.predict(x_te[valid_test_rows]).ravel()

    # Calculate the AUC score for the current task
    auc = roc_auc_score(y_test, p_test)
    auc_scores.append(auc)
    print(f"{target}: Test AUC = {auc:.5f}")

# Calculate and print the average AUC score
avg_auc = np.mean(auc_scores)
print(f"\nAverage Test AUC: {avg_auc:.5f}")
