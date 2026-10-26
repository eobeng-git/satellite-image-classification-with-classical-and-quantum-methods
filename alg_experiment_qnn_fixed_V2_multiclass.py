"""
Classification experiment with classical models + Improved PennyLane QNN
Flattens 2D image data to 1D tabular for standard CV
Uses pure default.qubit backend (no JAX/Lightning/AVX2 required)
V2 - With fixed QNN training and preprocessing
"""

# DEBUG PRINTS AT THE VERY BEGINNING
print("0. SCRIPT STARTED - BEFORE ANY IMPORTS")
import sys
sys.stdout.flush()
print("1. sys imported")
sys.stdout.flush()

"""
Module docstring
"""
print("2. About to import pathlib")
sys.stdout.flush()
from pathlib import Path
print("3. pathlib imported")
sys.stdout.flush()

print("4. About to import pandas")
sys.stdout.flush()
import pandas as pd
print("5. pandas imported")
sys.stdout.flush()

print("6. About to import numpy")
sys.stdout.flush()
import numpy as np
print("7. numpy imported")
sys.stdout.flush()

print("8. About to import sklearn.svm")
sys.stdout.flush()
from sklearn.svm import SVC
print("9. sklearn.svm imported")
sys.stdout.flush()

print("10. About to import sklearn.neighbors")
sys.stdout.flush()
from sklearn.neighbors import KNeighborsClassifier
print("11. sklearn.neighbors imported")
sys.stdout.flush()

print("12. About to import sklearn.naive_bayes")
sys.stdout.flush()
from sklearn.naive_bayes import GaussianNB
print("13. sklearn.naive_bayes imported")
sys.stdout.flush()

print("14. About to import sklearn.tree")
sys.stdout.flush()
from sklearn.tree import DecisionTreeClassifier
print("15. sklearn.tree imported")
sys.stdout.flush()

print("16. About to import sklearn.model_selection")
sys.stdout.flush()
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
print("17. sklearn.model_selection imported")
sys.stdout.flush()

print("18. About to import sklearn.preprocessing")
sys.stdout.flush()
from sklearn.preprocessing import StandardScaler
print("19. sklearn.preprocessing imported")
sys.stdout.flush()

print("20. About to import sklearn.pipeline")
sys.stdout.flush()
from sklearn.pipeline import Pipeline
print("21. sklearn.pipeline imported")
sys.stdout.flush()

print("22. About to import warnings")
sys.stdout.flush()
import warnings
print("23. warnings imported")
sys.stdout.flush()

print("24. About to import pennylane")
sys.stdout.flush()
import pennylane as qml
print("25. pennylane imported")
sys.stdout.flush()

print("26. About to import pennylane.numpy")
sys.stdout.flush()
from pennylane import numpy as np_pennylane
print("27. pennylane.numpy imported")
sys.stdout.flush()

print("28. About to import time and psutil for monitoring")
sys.stdout.flush()
import time
import psutil
import os
print("29. time and psutil imported")
sys.stdout.flush()

print("30. ALL IMPORTS COMPLETED SUCCESSFULLY!")
sys.stdout.flush()

# Create log file to track progress
with open('/NFSHOME/eobeng/src.disim/job-progress.log', 'w') as f:
    f.write(f"Job started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_message(msg):
    """Helper function to log messages"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open('/NFSHOME/eobeng/src.disim/job-progress.log', 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_message("Starting experiment V2 with improved QNN")

warnings.simplefilter(action="ignore", category=FutureWarning)

# =============================================================================
# CLASSICAL MODELS
# =============================================================================

log_message("Defining classical models...")

MODELS = {
    "svc": {
        "estimator": SVC(),
        "param_grid": {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]},
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [1, 3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
        },
    },
    "nb": {"estimator": GaussianNB(), "param_grid": {}},
    "dtree": {
        "estimator": DecisionTreeClassifier(),
        "param_grid": {
            "max_depth": [None, 2, 3, 5, 10],
            "min_samples_split": [2, 3, 4],
        },
    },
}

log_message("Classical models defined")

# =============================================================================
# IMPROVED PENNYLANE QNN (with proper training)
# =============================================================================

log_message("Defining improved PennyQNN class...")

class PennyQNN:
    """
    Improved PennyLane QNN classifier with proper training and preprocessing
    """
    def __init__(self, n_qubits=4, n_layers=2, epochs=10, learning_rate=0.1, batch_size=32):
        """
        Initialize the QNN classifier
        
        Args:
            n_qubits: Number of qubits to use
            n_layers: Number of variational layers
            epochs: Number of training epochs
            learning_rate: Step size for gradient descent
            batch_size: Mini-batch size for training
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        log_message(f"Initializing QNN with {n_qubits} qubits, {n_layers} layers")
        
        # Initialize weights randomly
        weight_shape = (n_layers, n_qubits, 2)  # Using 2 parameters per qubit per layer (RY, RZ)
        self.weights = np_pennylane.random.random(weight_shape, requires_grad=True)
        
        # Initialize device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.circuit = self._create_circuit()
        
        self.trained = False
        self.X_min = None
        self.X_max = None
        self.n_classes = None  # inferred from y at fit time (multiclass support)
        self.classes_ = None   # original class codes, sorted (sklearn convention)
        
        log_message("QNN initialized successfully")
    
    def _create_circuit(self):
        """Create the quantum circuit"""
        
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Normalize inputs to [0, 2π] for angle embedding
            inputs = inputs * 2 * np.pi
            
            # Encode classical data into quantum states
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
            
            # Variational layers
            for layer in range(self.n_layers):
                # Rotations
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Additional CNOT to create a cycle
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measure the full computational-basis distribution (2**n_qubits outcomes).
            # The first n_classes bins are interpreted as (unnormalized) class scores,
            # which supports an arbitrary number of classes (up to 2**n_qubits)
            # instead of a single binary readout.
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit

    def _class_probs(self, x, weights):
        """
        Run the circuit and return a normalized probability vector over
        self.n_classes classes, taken from the first n_classes bins of the
        full 2**n_qubits computational-basis distribution.
        """
        probs = self.circuit(x, weights)[: self.n_classes]
        probs = probs / (np_pennylane.sum(probs) + 1e-8)
        return probs
    
    def _preprocess_features(self, X):
        """
        Preprocess features for quantum circuit:
        1. Normalize to [0, 1] range
        2. Reduce dimensions if necessary
        """
        X = np.array(X)
        
        # Handle 1D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Store min/max for consistent preprocessing
        if self.X_min is None and self.X_max is None:
            self.X_min = X.min(axis=0)
            self.X_max = X.max(axis=0)
        
        # Normalize to [0, 1]
        X_normalized = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        # Handle feature dimension
        n_features = X.shape[1]
        X_processed = []
        
        for x in X_normalized:
            if n_features > self.n_qubits:
                # Reduce features by averaging groups
                group_size = n_features // self.n_qubits
                reduced_x = []
                for i in range(self.n_qubits):
                    start = i * group_size
                    end = start + group_size if i < self.n_qubits - 1 else n_features
                    reduced_x.append(np.mean(x[start:end]))
                X_processed.append(reduced_x)
            elif n_features < self.n_qubits:
                # Pad with zeros if not enough features
                padded = np.zeros(self.n_qubits)
                padded[:n_features] = x
                X_processed.append(padded)
            else:
                X_processed.append(x)
        
        return np_pennylane.array(X_processed)
    
    def fit(self, X, y):
        """
        Train the QNN using gradient descent
        
        Args:
            X: Training features
            y: Training labels (integer-encoded, 0..n_classes-1). Supports
               any number of classes, not just binary.
        """
        log_message(f"Starting QNN training on {len(X)} samples")
        
        # Infer classes from the training labels. Sentinel-2 class codes are
        # NOT contiguous (e.g. {0, 62, 73, 82, ...}), so we map them to
        # contiguous indices 0..n_classes-1 for training, and store the
        # original codes in self.classes_ (sklearn convention) to decode
        # predictions back to the real labels.
        y_arr = np.asarray(y)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        self.n_classes = len(self.classes_)
        max_classes = 2 ** self.n_qubits
        if self.n_classes > max_classes:
            raise ValueError(
                f"PennyQNN has {self.n_qubits} qubits, which can only distinguish "
                f"{max_classes} classes, but {self.n_classes} classes were found. "
                f"Increase n_qubits to at least {int(np.ceil(np.log2(self.n_classes)))}."
            )
        log_message(f"Detected {self.n_classes} classes {list(self.classes_)} (capacity: {max_classes})")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # One-hot encode the contiguous label indices for cross-entropy training
        y_onehot = np.eye(self.n_classes)[y_idx]
        y_onehot = np_pennylane.array(y_onehot, requires_grad=False)
        
        # Keep integer index labels (for accuracy tracking via self.score, which
        # compares against the original y outside this method)
        y = np_pennylane.array(y_arr, requires_grad=False)
        
        # Create optimizer
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        n_samples = len(X_processed)
        n_batches = max(1, n_samples // self.batch_size)
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_processed[batch_indices]
                y_onehot_batch = y_onehot[batch_indices]
                
                # Define cost function for this batch (multiclass cross-entropy)
                def cost(weights):
                    losses = []
                    for x, target in zip(X_batch, y_onehot_batch):
                        probs = self._class_probs(x, weights)
                        sample_loss = -np_pennylane.sum(target * np_pennylane.log(probs + 1e-8))
                        losses.append(sample_loss)
                    loss = np_pennylane.mean(np_pennylane.array(losses))
                    return loss
                
                # Update weights
                self.weights = opt.step(cost, self.weights)
                
                # Calculate batch loss
                batch_loss = float(cost(self.weights))
                epoch_loss += batch_loss
            
            # Calculate epoch metrics
            train_acc = self.score(X, y)
            avg_loss = epoch_loss / n_batches
            
            log_message(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Accuracy: {train_acc:.4f}")
        
        self.trained = True
        log_message("QNN training completed")
        
        return self
    
    def predict(self, X):
        """
        Predict labels for input data
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted class labels (0..n_classes-1)
        """
        if not self.trained:
            log_message("WARNING: Using untrained QNN for predictions!")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Make predictions
        predictions = []
        n_samples = len(X_processed)
        
        log_message(f"Making predictions on {n_samples} samples...")
        
        for i, x in enumerate(X_processed):
            if i > 0 and i % 100 == 0:
                log_message(f"  Progress: {i}/{n_samples} samples")
            
            # Multiclass: predicted class index is the argmax over the first
            # n_classes bins of the measured computational-basis distribution;
            # decode back to the original class code (e.g. 73, 82, ...)
            probs = self._class_probs(x, self.weights)
            pred_idx = int(np.argmax(np.array(probs)))
            pred = self.classes_[pred_idx]
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Get probability-like scores for predictions
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, n_classes) with per-class probabilities
            (rows sum to ~1). Column j corresponds to class self.classes_[j],
            analogous to sklearn's predict_proba.
        """
        if not self.trained:
            log_message("WARNING: Using untrained QNN for predictions!")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Get per-class probability vectors
        scores = []
        for x in X_processed:
            probs = self._class_probs(x, self.weights)
            scores.append(np.array(probs))
        
        return np.array(scores)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Accuracy score (fraction correct)
        """
        preds = self.predict(X)
        return np.mean(preds == y)

# Add QNN to models dictionary
MODELS["penny_qnn"] = {
    "estimator": PennyQNN(n_qubits=4, n_layers=2, epochs=10, learning_rate=0.1),
    "param_grid": {}  # Can add parameters to tune later
}

log_message("Improved QNN class defined and added to MODELS")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def test_classifier(X_train, y_train, X_test, y_test, k_internal=3, scoring="accuracy", scale_data=True, n_jobs=1, **model):
    estimator = model["estimator"]
    param_grid = model.get("param_grid", {})
    
    log_message(f"Testing classifier: {estimator.__class__.__name__}")
    log_message(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    if scale_data:
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    else:
        pipeline = estimator

    if param_grid:
        log_message(f"Running GridSearchCV with {len(param_grid)} parameters")
        gscv = GridSearchCV(pipeline, param_grid, cv=k_internal, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=1)
        gscv.fit(X_train, y_train)
        best_model = gscv.best_estimator_
        train_score = gscv.best_score_
        log_message(f"Best parameters: {gscv.best_params_}")
    else:
        log_message("No grid search - fitting directly")
        best_model = pipeline
        best_model.fit(X_train, y_train)
        train_score = best_model.score(X_train, y_train)

    test_score = best_model.score(X_test, y_test)
    log_message(f"Train score: {train_score:.4f}, Test score: {test_score:.4f}")
    
    return pd.DataFrame({"train_score": [train_score], "test_score": [test_score]})


def perform_experiment(X, y, instance_index=42, name="experiment", k_external=3, k_internal=2, scoring="accuracy", path="results/", scale_data=True, n_jobs=1):
    log_message(f"Starting experiment with {k_external} external folds, {k_internal} internal folds")
    log_message(f"Total samples: {len(X)}, Classes: {np.unique(y)}")
    
    skf = StratifiedKFold(n_splits=k_external, shuffle=True, random_state=instance_index)
    results = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        log_message(f"\n{'='*50}")
        log_message(f"FOLD {fold_idx}/{k_external}")
        log_message(f"{'='*50}")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        log_message(f"Fold {fold_idx} - Train: {len(X_train)}, Test: {len(X_test)}")
        
        fold_results = []

        for model_name, model in MODELS.items():
            log_message(f"\n--- Testing model: {model_name} ---")
            try:
                start_time = time.time()
                df = test_classifier(
                    X_train, y_train, X_test, y_test,
                    k_internal=k_internal, scoring=scoring, 
                    scale_data=scale_data, n_jobs=n_jobs, **model
                )
                elapsed = time.time() - start_time
                
                df["cls_name"] = model_name
                df["fold"] = fold_idx
                df["time_seconds"] = elapsed
                fold_results.append(df)
                
                log_message(f"✓ {model_name} completed in {elapsed:.2f} seconds")
            except Exception as e:
                log_message(f"✗ ERROR with {model_name}: {str(e)}")
                # Create error row
                error_df = pd.DataFrame({
                    "train_score": [np.nan], 
                    "test_score": [np.nan],
                    "cls_name": [model_name],
                    "fold": [fold_idx],
                    "time_seconds": [0],
                    "error": [str(e)]
                })
                fold_results.append(error_df)

        results.append(pd.concat(fold_results))
        
        # Log memory usage
        process = psutil.Process(os.getpid())
        log_message(f"Memory usage after fold {fold_idx}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    df_results = pd.concat(results, ignore_index=True)
    
    # Save results
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        fname = Path(f"{path}{name}_{scoring}_{instance_index}.pkl")
        csv_fname = Path(f"{path}{name}_{scoring}_{instance_index}.csv")
        
        df_results.to_pickle(fname)
        df_results.to_csv(csv_fname)
        log_message(f"\nResults saved to: {fname} and {csv_fname}")
        
        # Print summary
        summary = df_results.groupby('cls_name')[['train_score', 'test_score', 'time_seconds']].agg(['mean', 'std'])
        log_message(f"\n{'='*50}")
        log_message("FINAL RESULTS SUMMARY:")
        log_message(f"{'='*50}")
        log_message(str(summary))

    return df_results


# =============================================================================
# MAIN: LOAD DATA & RUN
# =============================================================================

if __name__ == "__main__":
    try:
        log_message("="*50)
        log_message("STARTING EXPERIMENT V2")
        log_message("="*50)
        
        # Load data
        log_message("Loading data...")
        data_path = "data/sentinel.npz"
        data = np.load(data_path)
        
        # Print data shapes
        log_message(f"Data keys: {list(data.keys())}")
        log_message(f"BANDS shape: {data['bands'].shape}")
        log_message(f"CLASSES shape: {data['classes'].shape}")
        
        # Flatten data
        X = data["bands"].reshape(-1, data["bands"].shape[-1])
        y = data["classes"].ravel()
        
        log_message(f"Flattened data: {X.shape[0]} samples, {X.shape[1]} features")
        log_message(f"Unique classes: {np.unique(y)}")
        log_message(f"Class distribution: {np.bincount(y)}")
        
        # USE FULL DATASET
        log_message("\n" + "="*50)
        log_message("USING FULL DATASET")
        log_message("="*50)

        # Use the full dataset (no sampling)
        X_sample = X
        y_sample = y
        log_message(f"Using FULL dataset: {len(X_sample)} samples")
        
        log_message(f"Using {len(X_sample)} samples for testing")
        log_message(f"Sample class distribution: {np.bincount(y_sample)}")
        
        # Run experiment on sample
        log_message("\n" + "="*50)
        log_message("RUNNING EXPERIMENT")
        log_message("="*50)
        
        start_time = time.time()
        
        results = perform_experiment(
            X=X_sample,
            y=y_sample,
            instance_index=42,
            name="qnn_full_experiment",
            k_external=3,  # Reduced folds
            k_internal=2,   # Reduced internal CV
            scoring="accuracy",
            path="results_qnn_full/",
            scale_data=True,
            n_jobs=1
        )
        
        total_time = time.time() - start_time
        log_message(f"\nTotal experiment time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")
        log_message("="*50)
        log_message("EXPERIMENT COMPLETED SUCCESSFULLY")
        log_message("="*50)
        
    except Exception as e:
        log_message(f"FATAL ERROR: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

