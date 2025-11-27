
```
Create a memory-efficient and fast-executing Jupyter notebook `notebooks/04_deep_learning_lstm.ipynb` for the Rossmann Sales Forecasting project with the following requirements:

PROJECT CONTEXT:
- Predicting daily sales for Rossmann pharmaceutical stores 6 weeks ahead using LSTM
- Running on LOCAL MACHINE (not Colab)
- Optimize for typical laptop/desktop specs (8-16GB RAM, possible GPU)
- Compare LSTM performance with ML models from 03_ml_modeling.ipynb

CRITICAL PERFORMANCE REQUIREMENTS:
⚡ MEMORY OPTIMIZATION:
- Use data sampling strategies for large datasets
- Implement batch processing
- Use generators for data loading (yield, not loading all at once)
- Clear memory explicitly with gc.collect() and del
- Use float32 instead of float64
- Process stores in batches, not all at once

⚡ SPEED OPTIMIZATION:
- Detect and use GPU if available (NVIDIA CUDA or Apple Silicon)
- Implement early stopping to avoid overtraining
- Use smaller model architecture (2 layers max as per requirements)
- Reduce sequence length if needed
- Use efficient data pipeline (tf.data or DataLoader)
- Vectorized operations only, no Python loops

NOTEBOOK STRUCTURE:

1. INTRODUCTION & SETUP
   - Markdown explaining LSTM for time series forecasting
   - Why LSTM for sales prediction vs traditional ML
   - Memory and speed optimization strategy overview
   - Import libraries (tensorflow/keras OR pytorch, pandas, numpy, matplotlib, gc, datetime, logging)
   - Detect local hardware and configure:
     ```python
     import sys
     import platform
     
     print(f"Python version: {sys.version}")
     print(f"Platform: {platform.platform()}")
     
     # For TensorFlow
     import tensorflow as tf
     gpus = tf.config.list_physical_devices('GPU')
     print(f"GPUs Available: {len(gpus)}")
     if gpus:
         try:
             # Enable memory growth to avoid allocating all GPU memory
             for gpu in gpus:
                 tf.config.experimental.set_memory_growth(gpu, True)
             print("GPU memory growth enabled")
         except RuntimeError as e:
             print(e)
     
     # For PyTorch
     import torch
     print(f"PyTorch GPU Available: {torch.cuda.is_available()}")
     if torch.cuda.is_available():
         print(f"GPU Device: {torch.cuda.get_device_name(0)}")
     # Check for Apple Silicon
     if torch.backends.mps.is_available():
         print("Apple Silicon MPS Available")
         device = torch.device("mps")
     elif torch.cuda.is_available():
         device = torch.device("cuda")
     else:
         device = torch.device("cpu")
     print(f"Using device: {device}")
     ```
   - Set up logging to logs/lstm_training.log
   - Set random seeds for reproducibility (42)
   - Configure matplotlib for non-interactive backend if needed

2. SYSTEM RESOURCE MONITORING
   
   Create utility functions to monitor local resources:
   ```python
   import psutil
   import os
   
   def get_memory_usage():
       """Get current process memory usage in MB"""
       process = psutil.Process(os.getpid())
       mem_mb = process.memory_info().rss / 1024 / 1024
       return mem_mb
   
   def get_available_memory():
       """Get available system memory in MB"""
       mem = psutil.virtual_memory()
       return mem.available / 1024 / 1024
   
   def log_memory_usage(stage):
       """Log memory usage at different stages"""
       used = get_memory_usage()
       available = get_available_memory()
       print(f"[{stage}] Memory used: {used:.2f} MB | Available: {available:.2f} MB")
       logging.info(f"[{stage}] Memory used: {used:.2f} MB | Available: {available:.2f} MB")
   
   # Monitor at start
   log_memory_usage("Notebook Start")
   ```

3. EFFICIENT DATA LOADING
   
   3.1 Sample Data Strategy:
   - Load processed data from data/processed/
   - **CRITICAL**: Don't load all stores at once
   - Sample strategy options:
     * Option A: Select top 100 stores by total sales
     * Option B: Stratified sample across store types
     * Option C: Random sample of 50-100 stores
   - Justify sampling approach (demonstrate on sample, then scale if needed)
   - Display memory usage before/after sampling
   
   3.2 Memory-Efficient Loading:
   ```python
   import pandas as pd
   import gc
   
   # Check file size first
   import os
   file_path = 'data/processed/train_processed.csv'
   file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
   print(f"File size: {file_size_mb:.2f} MB")
   
   log_memory_usage("Before data load")
   
   # Load only necessary columns and use efficient dtypes
   needed_cols = ['Date', 'Store', 'Sales'] + top_features[:10]  # Top 10 features only
   
   # For large files, use chunked reading
   if file_size_mb > 500:
       chunks = pd.read_csv(file_path, 
                           chunksize=100000,
                           usecols=needed_cols,
                           dtype={'Sales': 'float32'})  # Use float32 not float64
       
       # Select only needed stores
       selected_stores = pd.read_csv(file_path, usecols=['Store'], 
                                    nrows=1000)['Store'].unique()[:100]
       
       sampled_data = pd.concat([chunk[chunk['Store'].isin(selected_stores)] 
                                 for chunk in chunks])
       del chunks
   else:
       # Load all at once if file is small
       sampled_data = pd.read_csv(file_path, 
                                  usecols=needed_cols,
                                  dtype={'Sales': 'float32'})
       # Sample stores
       selected_stores = sampled_data['Store'].unique()[:100]
       sampled_data = sampled_data[sampled_data['Store'].isin(selected_stores)]
   
   log_memory_usage("After data load")
   
   # Clean up
   gc.collect()
   log_memory_usage("After garbage collection")
   ```
   
   3.3 Feature Selection:
   - Select only top 10-15 most important features (from ML notebook)
   - Reduce dimensionality to speed up training
   - Show memory savings from feature reduction
   - Convert all float64 to float32

4. TIME SERIES STATIONARITY CHECK
   
   4.1 Efficient Stationarity Testing:
   - Test on sample of 5-10 representative stores (not all)
   - Use Augmented Dickey-Fuller test from statsmodels
   - Create function to test stationarity efficiently:
   ```python
   from statsmodels.tsa.stattools import adfuller
   
   def check_stationarity_sample(data, n_stores=5):
       """Test stationarity on sample stores only"""
       sample_stores = data['Store'].unique()[:n_stores]
       results = {}
       
       for store in sample_stores:
           store_data = data[data['Store']==store].sort_values('Date')['Sales']
           adf_result = adfuller(store_data.dropna())
           results[store] = {
               'adf_statistic': adf_result[0],
               'p_value': adf_result[1],
               'is_stationary': adf_result[1] < 0.05
           }
       
       return pd.DataFrame(results).T
   
   # Run test
   stationarity_results = check_stationarity_sample(sampled_data, n_stores=5)
   print(stationarity_results)
   ```
   - Visualize time series for sample stores only (3-5 stores max)
   - Document findings
   
   4.2 Differencing (if needed):
   - Apply differencing only if non-stationary
   - Use vectorized operations:
   ```python
   # Efficient differencing with groupby
   sampled_data = sampled_data.sort_values(['Store', 'Date'])
   sampled_data['Sales_Diff'] = sampled_data.groupby('Store')['Sales'].diff()
   
   # Store original first value for each store (needed for inverse transform)
   first_values = sampled_data.groupby('Store')['Sales'].first()
   ```
   - Store differencing parameters for inverse transformation later

5. AUTOCORRELATION ANALYSIS
   
   - Compute ACF and PACF on sample data only (not full dataset)
   - Plot for 2-3 representative stores
   ```python
   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   
   # Analyze only 3 stores to save memory
   fig, axes = plt.subplots(3, 2, figsize=(12, 10))
   
   for idx, store in enumerate(selected_stores[:3]):
       store_data = sampled_data[sampled_data['Store']==store]['Sales'].dropna()
       
       plot_acf(store_data, lags=30, ax=axes[idx, 0])
       axes[idx, 0].set_title(f'Store {store} - ACF')
       
       plot_pacf(store_data, lags=30, ax=axes[idx, 1])
       axes[idx, 1].set_title(f'Store {store} - PACF')
   
   plt.tight_layout()
   plt.savefig('reports/acf_pacf_analysis.png', dpi=100, bbox_inches='tight')
   plt.show()
   
   # Clear figure from memory
   plt.close()
   ```
   - Determine optimal sequence length (lookback window)
   - Recommend: Use 7-14 days lookback (not 30+) to save memory
   - Document findings

6. CREATE SUPERVISED LEARNING DATA (MEMORY-EFFICIENT)
   
   6.1 Efficient Sliding Window Implementation:
   ```python
   import numpy as np
   from typing import Generator, Tuple
   
   def create_sequences_efficient(data, store_id, feature_cols, 
                                  lookback=7, forecast_horizon=1):
       """
       Memory-efficient sequence creation for single store
       Returns numpy arrays, not generator (faster for training)
       """
       store_data = data[data['Store']==store_id].sort_values('Date')
       
       # Convert to numpy for speed
       sales = store_data['Sales'].values.astype('float32')
       features = store_data[feature_cols].values.astype('float32')
       
       n_sequences = len(sales) - lookback - forecast_horizon + 1
       
       if n_sequences <= 0:
           return None, None
       
       # Pre-allocate arrays (faster than appending)
       X = np.zeros((n_sequences, lookback, len(feature_cols)), dtype='float32')
       y = np.zeros((n_sequences, forecast_horizon), dtype='float32')
       
       for i in range(n_sequences):
           X[i] = features[i:i+lookback]
           y[i] = sales[i+lookback:i+lookback+forecast_horizon]
       
       return X, y
   
   # Process stores in batches to manage memory
   def create_dataset_batched(data, stores, feature_cols, 
                             lookback=7, batch_size=20):
       """
       Create dataset by processing stores in batches
       """
       X_list = []
       y_list = []
       
       for i in range(0, len(stores), batch_size):
           batch_stores = stores[i:i+batch_size]
           print(f"Processing stores {i} to {i+len(batch_stores)}...")
           
           for store in batch_stores:
               X_store, y_store = create_sequences_efficient(
                   data, store, feature_cols, lookback
               )
               if X_store is not None:
                   X_list.append(X_store)
                   y_list.append(y_store)
           
           # Log memory after each batch
           log_memory_usage(f"Batch {i//batch_size + 1}")
       
       # Concatenate all sequences
       X_all = np.concatenate(X_list, axis=0)
       y_all = np.concatenate(y_list, axis=0)
       
       # Clean up
       del X_list, y_list
       gc.collect()
       
       return X_all, y_all
   ```
   
   6.2 Create Train/Test Split:
   ```python
   # Define parameters
   LOOKBACK = 7  # Use past 7 days
   FORECAST_HORIZON = 1  # Predict next day
   
   # Split data temporally (last 6 weeks for test)
   test_weeks = 6
   test_days = test_weeks * 7
   
   train_data = sampled_data.groupby('Store').apply(
       lambda x: x.sort_values('Date').iloc[:-test_days]
   ).reset_index(drop=True)
   
   test_data = sampled_data.groupby('Store').apply(
       lambda x: x.sort_values('Date').iloc[-test_days:]
   ).reset_index(drop=True)
   
   print(f"Train date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
   print(f"Test date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
   
   # Create sequences
   log_memory_usage("Before sequence creation")
   
   X_train, y_train = create_dataset_batched(
       train_data, selected_stores, feature_cols, 
       lookback=LOOKBACK, batch_size=20
   )
   
   X_test, y_test = create_dataset_batched(
       test_data, selected_stores, feature_cols,
       lookback=LOOKBACK, batch_size=20
   )
   
   log_memory_usage("After sequence creation")
   
   print(f"X_train shape: {X_train.shape}")
   print(f"y_train shape: {y_train.shape}")
   print(f"X_test shape: {X_test.shape}")
   print(f"y_test shape: {y_test.shape}")
   
   # Clean up original dataframes if not needed
   del train_data, test_data
   gc.collect()
   ```

7. DATA SCALING (EFFICIENT)
   
   ```python
   from sklearn.preprocessing import MinMaxScaler
   import joblib
   
   # Scale to (-1, 1) range for LSTM
   scaler_X = MinMaxScaler(feature_range=(-1, 1))
   scaler_y = MinMaxScaler(feature_range=(-1, 1))
   
   # Reshape for scaling
   n_samples, n_timesteps, n_features = X_train.shape
   
   # Scale features
   X_train_reshaped = X_train.reshape(-1, n_features)
   X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
   X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
   
   # Scale test set
   X_test_reshaped = X_test.reshape(-1, n_features)
   X_test_scaled = scaler_X.transform(X_test_reshaped)
   X_test_scaled = X_test_scaled.reshape(X_test.shape)
   
   # Scale target
   y_train_scaled = scaler_y.fit_transform(y_train)
   y_test_scaled = scaler_y.transform(y_test)
   
   # Save scalers for later use
   joblib.dump(scaler_X, 'models/lstm_scaler_X.pkl')
   joblib.dump(scaler_y, 'models/lstm_scaler_y.pkl')
   
   print("Scaling complete and scalers saved")
   log_memory_usage("After scaling")
   
   # Clean up unscaled data
   del X_train_reshaped, X_test_reshaped
   gc.collect()
   ```

8. BUILD LIGHTWEIGHT LSTM MODEL
   
   8.1 Model Architecture (KEEP IT SIMPLE!):
   ```python
   # Choose framework based on availability
   USE_TENSORFLOW = True  # Set to False to use PyTorch
   
   if USE_TENSORFLOW:
       from tensorflow.keras.models import Sequential
       from tensorflow.keras.layers import LSTM, Dense, Dropout
       from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
       from tensorflow.keras.optimizers import Adam
       
       def build_efficient_lstm(n_timesteps, n_features):
           """
           Lightweight LSTM model - 2 layers as per requirements
           Small hidden units to run fast on local machine
           """
           model = Sequential([
               # Layer 1: First LSTM layer
               LSTM(32, 
                    activation='tanh', 
                    return_sequences=True,
                    input_shape=(n_timesteps, n_features),
                    name='lstm_1'),
               Dropout(0.2, name='dropout_1'),
               
               # Layer 2: Second LSTM layer
               LSTM(16, 
                    activation='tanh',
                    return_sequences=False,
                    name='lstm_2'),
               Dropout(0.2, name='dropout_2'),
               
               # Output layer
               Dense(1, name='output')
           ])
           
           # Compile with efficient optimizer
           optimizer = Adam(learning_rate=0.001)
           model.compile(
               optimizer=optimizer, 
               loss='mse',
               metrics=['mae']
           )
           
           return model
       
   else:  # PyTorch
       import torch
       import torch.nn as nn
       
       class EfficientLSTM(nn.Module):
           def __init__(self, input_size, hidden_size1=32, hidden_size2=16):
               super(EfficientLSTM, self).__init__()
               self.hidden_size1 = hidden_size1
               self.hidden_size2 = hidden_size2
               
               # Layer 1
               self.lstm1 = nn.LSTM(input_size, hidden_size1, 
                                   batch_first=True)
               self.dropout1 = nn.Dropout(0.2)
               
               # Layer 2
               self.lstm2 = nn.LSTM(hidden_size1, hidden_size2,
                                   batch_first=True)
               self.dropout2 = nn.Dropout(0.2)
               
               # Output layer
               self.fc = nn.Linear(hidden_size2, 1)
           
           def forward(self, x):
               # Layer 1
               lstm1_out, _ = self.lstm1(x)
               lstm1_out = self.dropout1(lstm1_out)
               
               # Layer 2
               lstm2_out, _ = self.lstm2(lstm1_out)
               lstm2_out = self.dropout2(lstm2_out)
               
               # Take last timestep output
               last_output = lstm2_out[:, -1, :]
               
               # Output
               out = self.fc(last_output)
               return out
   ```
   
   8.2 Initialize Model:
   ```python
   # Build model
   _, n_timesteps, n_features = X_train_scaled.shape
   
   if USE_TENSORFLOW:
       model = build_efficient_lstm(n_timesteps, n_features)
       print(model.summary())
       
       # Calculate model size
       total_params = model.count_params()
       print(f"\nTotal parameters: {total_params:,}")
       print(f"Estimated model size: {total_params * 4 / 1024 / 1024:.2f} MB")
       
   else:
       model = EfficientLSTM(n_features).to(device)
       print(model)
       
       # Calculate model size
       total_params = sum(p.numel() for p in model.parameters())
       print(f"\nTotal parameters: {total_params:,}")
       print(f"Estimated model size: {total_params * 4 / 1024 / 1024:.2f} MB")
   ```

9. TRAINING STRATEGY (FAST & EFFICIENT)
   
   9.1 Training Configuration:
   ```python
   import time
   from datetime import datetime
   
   # Training hyperparameters
   BATCH_SIZE = 64          # Balance between speed and memory
   EPOCHS = 50              # Not too many, use early stopping
   VALIDATION_SPLIT = 0.2   # 20% of training data for validation
   
   if USE_TENSORFLOW:
       # Callbacks for efficient training
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       
       early_stop = EarlyStopping(
           monitor='val_loss',
           patience=7,              # Stop if no improvement for 7 epochs
           restore_best_weights=True,
           verbose=1
       )
       
       reduce_lr = ReduceLROnPlateau(
           monitor='val_loss',
           factor=0.5,              # Reduce LR by half
           patience=3,              # After 3 epochs without improvement
           min_lr=1e-6,
           verbose=1
       )
       
       checkpoint = ModelCheckpoint(
           f'models/lstm_best_{timestamp}.h5',
           save_best_only=True,
           monitor='val_loss',
           verbose=1
       )
       
       callbacks = [early_stop, reduce_lr, checkpoint]
   ```
   
   9.2 Training Execution (TensorFlow):
   ```python
   if USE_TENSORFLOW:
       print("Starting training...")
       print(f"Device: {tf.test.gpu_device_name() if tf.test.is_gpu_available() else 'CPU'}")
       
       log_memory_usage("Before training")
       start_time = time.time()
       
       # Train model
       history = model.fit(
           X_train_scaled, y_train_scaled,
           batch_size=BATCH_SIZE,
           epochs=EPOCHS,
           validation_split=VALIDATION_SPLIT,
           callbacks=callbacks,
           verbose=1
       )
       
       training_time = time.time() - start_time
       print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
       
       log_memory_usage("After training")
       
       # Save final model
       model.save(f'models/lstm_final_{timestamp}.h5')
       print(f"Model saved as lstm_final_{timestamp}.h5")
   ```
   
   9.3 Training Execution (PyTorch):
   ```python
   if not USE_TENSORFLOW:
       import torch.optim as optim
       from torch.utils.data import TensorDataset, DataLoader
       
       # Prepare data loaders
       train_dataset = TensorDataset(
           torch.FloatTensor(X_train_scaled),
           torch.FloatTensor(y_train_scaled)
       )
       
       # Split for validation
       val_size = int(0.2 * len(train_dataset))
       train_size = len(train_dataset) - val_size
       train_subset, val_subset = torch.utils.data.random_split(
           train_dataset, [train_size, val_size]
       )
       
       train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=2)
       val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
       
       # Loss and optimizer
       criterion = nn.MSELoss()
       optimizer = optim.Adam(model.parameters(), lr=0.001)
       scheduler = optim.lr_scheduler.ReduceLROnPlateau(
           optimizer, mode='min', factor=0.5, patience=3, verbose=True
       )
       
       # Training loop
       print("Starting training...")
       print(f"Device: {device}")
       
       log_memory_usage("Before training")
       start_time = time.time()
       
       train_losses = []
       val_losses = []
       best_val_loss = float('inf')
       patience_counter = 0
       
       for epoch in range(EPOCHS):
           # Training
           model.train()
           train_loss = 0
           for X_batch, y_batch in train_loader:
               X_batch, y_batch = X_batch.to(device), y_batch.to(device)
               
               optimizer.zero_grad()
               outputs = model(X_batch)
               loss = criterion(outputs, y_batch)
               loss.backward()
               optimizer.step()
               
               train_loss += loss.item()
           
           train_loss /= len(train_loader)
           train_losses.append(train_loss)
           
           # Validation
           model.eval()
           val_loss = 0
           with torch.no_grad():
               for X_batch, y_batch in val_loader:
                   X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                   outputs = model(X_batch)
                   loss = criterion(outputs, y_batch)
                   val_loss += loss.item()
           
           val_loss /= len(val_loader)
           val_losses.append(val_loss)
           
           # Learning rate scheduling
           scheduler.step(val_loss)
           
           # Early stopping
           if val_loss < best_val_loss:
               best_val_loss = val_loss
               patience_counter = 0
               # Save best model
               torch.save(model.state_dict(), f'models/lstm_best_{timestamp}.pt')
           else:
               patience_counter += 1
               if patience_counter >= 7:
                   print(f"Early stopping at epoch {epoch+1}")
                   break
           
           if (epoch + 1) % 5 == 0:
               print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
       
       training_time = time.time() - start_time
       print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
       
       log_memory_usage("After training")
       
       # Load best model
       model.load_state_dict(torch.load(f'models/lstm_best_{timestamp}.pt'))
   ```
   
   9.4 Plot Training History:
   ```python
   # Plot training history
   plt.figure(figsize=(12, 4))
   
   if USE_TENSORFLOW:
       # Loss
       plt.subplot(1, 2, 1)
       plt.plot(history.history['loss'], label='Train Loss')
       plt.plot(history.history['val_loss'], label='Val Loss')
       plt.title('Model Loss During Training')
       plt.xlabel('Epoch')
       plt.ylabel('Loss (MSE)')
       plt.legend()
       plt.grid(True)
       
       # MAE
       plt.subplot(1, 2, 2)
       plt.plot(history.history['mae'], label='Train MAE')
       plt.plot(history.history['val_mae'], label='Val MAE')
       plt.title('Model MAE During Training')
       plt.xlabel('Epoch')
       plt.ylabel('MAE')
       plt.legend()
       plt.grid(True)
   else:
       # Loss
       plt.subplot(1, 2, 1)
       plt.plot(train_losses, label='Train Loss')
       plt.plot(val_losses, label='Val Loss')
       plt.title('Model Loss During Training')
       plt.xlabel('Epoch')
       plt.ylabel('Loss (MSE)')
       plt.legend()
       plt.grid(True)
   
   plt.tight_layout()
   plt.savefig('reports/lstm_training_history.png', dpi=100, bbox_inches='tight')
   plt.show()
   
   # Clear figure
   plt.close()
   ```

10. MODEL EVALUATION (EFFICIENT)
    
    10.1 Generate Predictions:
    ```python
    print("Generating predictions...")
    log_memory_usage("Before prediction")
    
    if USE_TENSORFLOW:
        # Predict in batches
        y_pred_scaled = model.predict(X_test_scaled, batch_size=BATCH_SIZE, verbose=1)
        
    else:
        model.eval()
        predictions = []
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for X_batch, in test_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                predictions.append(pred.cpu().numpy())
        
        y_pred_scaled = np.concatenate(predictions, axis=0)
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test_scaled)
    
    print(f"Predictions shape: {y_pred.shape}")
    log_memory_usage("After prediction")
    ```
    
    10.2 Calculate Metrics:
    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0  # Avoid division by zero
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_metrics(y_true, y_pred, model_name="LSTM"):
        """Calculate all regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # RMSE as percentage of mean
        mean_sales = np.mean(y_true)
        rmse_pct = (rmse / mean_sales) * 100
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'RMSE%': rmse_pct,
            'MAPE': mape,
            'R²': r2
        }
        
        return metrics
    
    # Calculate metrics
    lstm_metrics = calculate_metrics(y_test_actual, y_pred, "LSTM")
    
    # Display metrics
    print("\n" + "="*60)
    print("LSTM MODEL PERFORMANCE METRICS")
    print("="*60)
    for metric, value in lstm_metrics.items():
        if metric == 'Model':
            continue
        print(f"{metric:12s}: {value:10.4f}")
    print("="*60)
    
    # Log metrics
    logging.info(f"LSTM Metrics: {lstm_metrics}")
    ```
    
    10.3 Visualizations:
    ```python
    # 1. Predictions vs Actual
    plt.figure(figsize=(14, 6))
    
    # Sample for visualization (first 500 points