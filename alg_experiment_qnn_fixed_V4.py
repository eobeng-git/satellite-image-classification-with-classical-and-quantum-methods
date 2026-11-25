"""
Classification experiment with classical models + Improved PennyLane QNN
Flattens 2D image data to 1D tabular for standard CV
Uses pure default.qubit backend (no JAX/Lightning/AVX2 required)
V2 - With fixed QNN training and preprocessing
V3 - PennyQNN state is reset at every fold (FIX 1) + convergence check (FIX 2); new log/results paths
V4 - Changes requested in the supervisor review (points 1-9):
     1  syntax: the script must compile (run `python -m py_compile` before starting; done in the .sh)
     2  independent initialisation per fold: the initial QNN weights of every fold are logged, saved and checked
     3  train_score = accuracy on the TRAINING set for every model; the grid-search score is stored as inner_cv_score
     4  mini-batches: ceil(n_samples / batch_size), so the last incomplete batch is used
     5  main comparison: tol=None, 10 fixed epochs (early stopping stays available, but is OFF)
     6  accuracy monitoring on a fixed stratified subset; full training accuracy once, after training
     7  F1, per-class precision/recall, balanced accuracy and a confusion matrix per fold
     8  seeds, fold indices, configurations, histories, epochs, predictions saved; failures are reported as failures
     9  hyper-parameter search, final fit and prediction times are measured separately

Metric references (as given in the review): precision/recall/F1 doi:10.1016/j.ipm.2009.03.002 ;
balanced accuracy doi:10.1109/ICPR.2010.764

Usage:  python alg_experiment_qnn_fixed_V4.py            full experiment
        python alg_experiment_qnn_fixed_V4.py --quick    small 3-fold check (3000 pixels, 2 QNN epochs)
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

import json
import platform
import traceback
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import confusion_matrix

# --quick = small three-fold check (see header). Separate log/results names so that nothing else is overwritten.
QUICK = "--quick" in sys.argv
LOG_FILE = 'job-progress-v4-quick.log' if QUICK else 'job-progress-v4.log'
RESULTS_PATH = "results_qnn_v4_quick/" if QUICK else "results_qnn_v4/"
with open(LOG_FILE, 'w') as f:
    f.write(f"Job started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_message(msg):
    """Helper function to log messages"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_message("Starting experiment V4 (supervisor review applied)" + (" - QUICK MODE" if QUICK else ""))

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
        "estimator": DecisionTreeClassifier(random_state=42),
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

class PennyQNN(ClassifierMixin, BaseEstimator):   # V4: proper scikit-learn estimator (needed by newer scikit-learn versions inside Pipeline)
    """
    Improved PennyLane QNN classifier with proper training and preprocessing
    """
    def __init__(self, n_qubits=4, n_layers=2, epochs=10, learning_rate=0.1, batch_size=32,
                 random_state=None, tol=None, patience=2, min_epochs=6, monitor_size=5000):   # FIX 1 (random_state), FIX 2 (tol, patience, min_epochs), V4 (monitor_size)
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
        self.random_state = random_state   # FIX 1: seed for initial weights and batch order
        self.tol = tol                     # FIX 2: convergence tolerance on the epoch loss (None = off)
        self.patience = patience           # epochs in a row below tol
        self.min_epochs = min_epochs       # never declare convergence before this epoch
        self.monitor_size = monitor_size   # V4: size of the fixed training subset used to monitor accuracy (None = all)
        self.epochs_completed_ = 0
        self.initial_weights_ = None
        self.final_weights_ = None
        
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
        
        # --- FIX 1 (the three folds): RESET so every fit() (= every CV fold) starts from scratch ---
        # Before: weights, X_min and X_max were kept from the previous fold (see the header of this file).
        rng = np.random.RandomState(self.random_state)
        self.weights = np_pennylane.array(rng.random_sample(self.weights.shape), requires_grad=True)
        self.X_min = None
        self.X_max = None
        self.trained = False
        self.loss_history_ = []
        log_message(f"QNN state reset: initial weights mean = {float(np.mean(self.weights)):.6f}")
        self.initial_weights_ = np.array(self.weights).copy()
        self.final_weights_ = None
        self.epochs_completed_ = 0
        log_message("QNN state reset: X_min/X_max, training history and trained flag cleared")
        log_message(f"QNN initial weights (start of this fit): {np.round(self.initial_weights_.flatten(), 6).tolist()}")
        
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
        # V4 (review point 4): ceil division, so the last incomplete mini-batch is used as well
        n_batches = max(1, (n_samples + self.batch_size - 1) // self.batch_size)
        
        # V4 (review point 6): fixed, class-stratified subset of the training data, used ONLY to monitor
        # the accuracy after every epoch (the full training accuracy is computed once, after training)
        monitor_idx = self._monitor_subset(y_idx, n_samples)
        log_message(f"Monitoring accuracy on a fixed stratified subset of {len(monitor_idx)} training samples")
        
        prev_loss, stall, converged = None, 0, False   # FIX 2: convergence monitoring
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = rng.permutation(n_samples)   # FIX 1 (was: np.random.permutation)
            
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
            
            # Epoch metrics: loss from the mini-batches; accuracy on the fixed monitoring subset only
            monitor_pred = self._predict_processed(X_processed[monitor_idx], verbose=False)
            monitor_acc = float(np.mean(monitor_pred == y_arr[monitor_idx]))
            avg_loss = epoch_loss / n_batches
            
            log_message(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Accuracy: {monitor_acc:.4f} "
                        f"| (accuracy on the fixed monitoring subset, n={len(monitor_idx)})")
            self.loss_history_.append((epoch + 1, avg_loss, monitor_acc))   # FIX 1 (used to verify the reset)
            self.epochs_completed_ = epoch + 1
            
            # --- FIX 2 (convergence): stop when the epoch loss has settled (only if tol is set) ---
            if self.tol is not None:
                stall = stall + 1 if (prev_loss is not None and abs(avg_loss - prev_loss) < self.tol) else 0
                prev_loss = avg_loss
                if epoch + 1 >= self.min_epochs and stall >= self.patience:
                    converged = True
                    log_message(f"CONVERGED at epoch {epoch+1}")
                    break
        
        self.trained = True
        self.final_weights_ = np.array(self.weights).copy()
        if self.tol is not None and not converged:
            log_message(f"NOT converged within the maximum of {self.epochs} epochs")
        log_message(f"QNN training completed after {self.epochs_completed_} of {self.epochs} epochs "
                    f"(early stopping {'ON' if self.tol is not None else 'OFF'}; the criterion monitors the stabilisation "
                    f"of the TRAINING loss, not validation performance)")
        
        return self
    
    def _monitor_subset(self, y_idx, n_samples):
        """Fixed, class-stratified subset of the training samples (same for every epoch)."""
        if self.monitor_size is None or self.monitor_size >= n_samples:
            return np.arange(n_samples)
        mrng = np.random.RandomState(None if self.random_state is None else self.random_state + 1)
        chosen = []
        for c in np.unique(y_idx):
            members = np.where(y_idx == c)[0]
            k = min(len(members), max(1, int(round(self.monitor_size * len(members) / n_samples))))
            chosen.append(mrng.choice(members, k, replace=False))
        return np.sort(np.concatenate(chosen))

    def _predict_processed(self, X_processed, verbose=True):
        """Predict from already pre-processed features. No warning: also used inside fit()."""
        predictions = []
        n_samples = len(X_processed)
        for i, x in enumerate(X_processed):
            if verbose and i > 0 and i % 10000 == 0:
                log_message(f"  Progress: {i}/{n_samples} samples")
            # Multiclass: argmax over the first n_classes bins of the measured distribution,
            # decoded back to the original class code (e.g. 73, 82, ...)
            probs = self._class_probs(x, self.weights)
            predictions.append(self.classes_[int(np.argmax(np.array(probs)))])
        return np.array(predictions)

    def predict(self, X):
        """
        Predict labels for input data

        Args:
            X: Input features

        Returns:
            Array of predicted class labels (original class codes)
        """
        if not self.trained:
            log_message("WARNING: predict() called on an untrained QNN!")
        X_processed = self._preprocess_features(X)
        log_message(f"Making predictions on {len(X_processed)} samples...")
        return self._predict_processed(X_processed)

    def __sklearn_is_fitted__(self):
        return bool(self.trained)

    def get_config(self):
        """Hyper-parameters of this model (saved with the results)."""
        return dict(model="PennyQNN", n_qubits=self.n_qubits, n_layers=self.n_layers, epochs=self.epochs,
                    learning_rate=self.learning_rate, batch_size=self.batch_size, random_state=self.random_state,
                    tol=self.tol, patience=self.patience, min_epochs=self.min_epochs,
                    monitor_size=self.monitor_size, optimizer="GradientDescentOptimizer",
                    loss="categorical cross-entropy")

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
    "estimator": PennyQNN(n_qubits=4, n_layers=2, epochs=10, learning_rate=0.1,
                         random_state=42, tol=None, patience=2, min_epochs=6, monitor_size=5000),
    "param_grid": {}  # Can add parameters to tune later
}

log_message("Improved QNN class defined and added to MODELS")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def stratified_subsample(X, y, n, min_per_class=6, seed=0):
    """Class-stratified random subsample of about n rows with at least `min_per_class` rows per class (used by --quick)."""
    rs = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    take = []
    for c, cnt in zip(classes, counts):
        k = min(cnt, max(min_per_class, int(round(n * cnt / len(y)))))
        take.append(rs.choice(np.where(y == c)[0], k, replace=False))
    idx = np.sort(np.concatenate(take))
    return X[idx], y[idx]


def metrics_from_confusion(cm):
    """
    Per-class and aggregate classification metrics from a confusion matrix
    (rows = true class, columns = predicted class).

    Undefined values are returned as NaN and are handled as follows:
      * precision of a class that was never predicted (TP + FP = 0)  -> NaN (undefined)
      * recall / F1 of a class with no true samples in the test fold  -> NaN (undefined)
    Aggregates are computed over the classes that are present in the test fold (support > 0):
      * balanced_accuracy = macro_recall = mean of the per-class recalls
      * macro_f1       : a present class that is never predicted has F1 = 0 (as in scikit-learn)
      * macro_precision: classes with undefined precision are EXCLUDED from the mean and their number is
                         reported (n_classes_undefined_precision); macro_precision_undefined_as_0 counts them
                         as 0.0 (the zero_division=0 convention of scikit-learn)
      * weighted_f1    : F1 weighted by the class support
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)       # true samples per class
    predicted = cm.sum(axis=0)     # predicted samples per class
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        denom = predicted + support            # 2TP + FP + FN = (TP+FP) + (TP+FN)
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    present = support > 0
    prec_defined = present & (predicted > 0)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "support": support, "predicted": predicted, "tp": tp,
        "accuracy": float(tp.sum() / cm.sum()),
        "balanced_accuracy": float(np.mean(recall[present])),
        "macro_recall": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "weighted_f1": float(np.sum(f1[present] * support[present]) / support[present].sum()),
        "macro_precision": float(np.mean(precision[prec_defined])) if prec_defined.any() else float("nan"),
        "macro_precision_undefined_as_0": float(np.mean(np.where(predicted[present] > 0, precision[present], 0.0))),
        "n_classes_undefined_precision": int((present & (predicted == 0)).sum()),
    }


def test_classifier(X_train, y_train, X_test, y_test, model, classes,
                    k_internal=2, scoring="accuracy", scale_data=True, n_jobs=1):
    """
    Evaluate one model on one outer fold.

    `model` is a dict {"estimator": ..., "param_grid": ...} (a plain argument: no **kwargs).
    Returns (row, artifacts): `row` = one result record, `artifacts` = predictions, confusion matrix,
    per-class metrics, configuration and training history of this model on this fold.

    Timings (seconds) are measured separately:
      search_time        hyper-parameter search (GridSearchCV; 0 for models without a grid)
      fit_time           final fit on the whole training fold
      predict_train_time prediction on the training fold
      predict_test_time  prediction on the test fold
    train_score is the accuracy on the TRAINING fold for every model (identical to best_model.score(X_train, y_train));
    the internal cross-validation score of the grid search is stored separately as inner_cv_score.
    """
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
        log_message(f"Running GridSearchCV with {len(param_grid)} parameters, {k_internal} internal folds")
        gscv = GridSearchCV(pipeline, param_grid, cv=k_internal, scoring=scoring, n_jobs=n_jobs, refit=False, verbose=1)
        t0 = time.time()
        gscv.fit(X_train, y_train)
        search_time = time.time() - t0
        best_params = dict(gscv.best_params_)
        inner_cv_score = float(gscv.best_score_)
        log_message(f"Best parameters: {best_params}  (inner CV {scoring}: {inner_cv_score:.4f}, search time {search_time:.1f}s)")
        # final model: best parameters, fitted on the whole training fold (this is the "refit" step, timed on its own)
        best_model = clone(pipeline).set_params(**best_params)
        t0 = time.time()
        best_model.fit(X_train, y_train)
        fit_time = time.time() - t0
    else:
        log_message("No grid search - fitting directly")
        search_time, best_params, inner_cv_score = 0.0, {}, float("nan")
        best_model = pipeline
        t0 = time.time()
        best_model.fit(X_train, y_train)
        fit_time = time.time() - t0

    t0 = time.time()
    y_pred_train = best_model.predict(X_train)
    predict_train_time = time.time() - t0
    t0 = time.time()
    y_pred_test = best_model.predict(X_test)
    predict_test_time = time.time() - t0

    train_score = float(np.mean(y_pred_train == y_train))
    test_score = float(np.mean(y_pred_test == y_test))

    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    m = metrics_from_confusion(cm)

    log_message(f"Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}, "
                f"Balanced accuracy: {m['balanced_accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}")
    if m["n_classes_undefined_precision"]:
        log_message(f"NOTE: {m['n_classes_undefined_precision']} class(es) were never predicted: their precision is "
                    f"undefined (NaN) and excluded from macro_precision")
    log_message(f"Times: search {search_time:.1f}s | fit {fit_time:.1f}s | predict train {predict_train_time:.1f}s | "
                f"predict test {predict_test_time:.1f}s")

    clf = best_model.steps[-1][1] if hasattr(best_model, "steps") else best_model
    config = clf.get_config() if hasattr(clf, "get_config") else clf.get_params(deep=False)

    row = {
        "train_score": train_score, "test_score": test_score, "inner_cv_score": inner_cv_score,
        "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
        "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
        "macro_precision_undefined_as_0": m["macro_precision_undefined_as_0"],
        "n_classes_undefined_precision": m["n_classes_undefined_precision"],
        "search_time": search_time, "fit_time": fit_time,
        "predict_train_time": predict_train_time, "predict_test_time": predict_test_time,
        "epochs_completed": getattr(clf, "epochs_completed_", float("nan")),
        "best_params": json.dumps(best_params, default=str, sort_keys=True),
    }
    per_class = pd.DataFrame({
        "class": classes, "support": m["support"], "predicted": m["predicted"], "tp": m["tp"],
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
    })
    artifacts = {
        "y_pred_test": y_pred_test, "confusion_matrix": cm, "per_class": per_class,
        "best_params": best_params, "config": config,
        "history": getattr(clf, "loss_history_", None),
        "initial_weights": getattr(clf, "initial_weights_", None),
        "final_weights": getattr(clf, "final_weights_", None),
    }
    return row, artifacts


def verify_qnn_initialization(initial_weights_by_fold, seed):
    """
    Review point 2: every fold must start from the seeded initial weights, i.e. no trained state is carried over.
    Logs the check and returns True if it passed.
    """
    if not initial_weights_by_fold:
        return True
    if seed is None:
        log_message("INIT CHECK skipped: the QNN has no random_state, so the initial weights cannot be reproduced")
        return True
    folds = sorted(initial_weights_by_fold)
    shape = initial_weights_by_fold[folds[0]].shape
    expected = np.random.RandomState(seed).random_sample(shape)
    ok = True
    for f in folds:
        w = initial_weights_by_fold[f]
        same = bool(np.array_equal(w, expected))
        diff_to_first = float(np.max(np.abs(w - initial_weights_by_fold[folds[0]])))
        log_message(f"INIT CHECK fold {f}: initial weights == seeded initialisation: {same} "
                    f"| max |difference| to fold {folds[0]}: {diff_to_first:.2e}")
        ok = ok and same
    log_message("INIT CHECK PASSED: every fold started from scratch" if ok else
                "INIT CHECK FAILED: a fold did not start from the seeded initialisation")
    return ok


def _json_safe(obj, fname):
    with open(fname, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def perform_experiment(X, y, instance_index=42, name="experiment", k_external=3, k_internal=2,
                       scoring="accuracy", path="results/", scale_data=True, n_jobs=1):
    import sklearn
    log_message(f"Starting experiment with {k_external} external folds, {k_internal} internal folds")
    log_message(f"Total samples: {len(X)}, Classes: {np.unique(y)}")

    classes = np.unique(y)
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=k_external, shuffle=True, random_state=instance_index)
    splits = list(skf.split(X, y))
    np.savez_compressed(out_dir / f"fold_indices_seed{instance_index}.npz",
                        **{f"train_fold{i}": tr for i, (tr, te) in enumerate(splits, 1)},
                        **{f"test_fold{i}": te for i, (tr, te) in enumerate(splits, 1)})

    metadata = {
        "script": os.path.basename(__file__), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "quick_mode": QUICK,
        "seeds": {"outer_split_random_state": instance_index,
                  "qnn_random_state": getattr(MODELS.get("penny_qnn", {}).get("estimator"), "random_state", None)},
        "settings": {"k_external": k_external, "k_internal": k_internal, "scoring": scoring,
                     "scale_data": scale_data, "n_jobs": n_jobs},
        "data": {"n_samples": int(len(X)), "n_features": int(X.shape[1]), "classes": classes.tolist()},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "pennylane": getattr(qml, "__version__", "unknown")},
        "models": {n: {"class": m["estimator"].__class__.__name__, "param_grid": m.get("param_grid", {}),
                       "config": (m["estimator"].get_config() if hasattr(m["estimator"], "get_config")
                                  else m["estimator"].get_params(deep=False))} for n, m in MODELS.items()},
        "folds": {},
    }

    rows, per_class_frames, failures, qnn_initial = [], [], [], {}

    def write_results(final=False):
        df = pd.DataFrame(rows)
        suffix = "" if final else "_partial"
        df.to_pickle(out_dir / f"{name}_{scoring}_{instance_index}{suffix}.pkl")
        df.to_csv(out_dir / f"{name}_{scoring}_{instance_index}{suffix}.csv")
        if per_class_frames:
            pd.concat(per_class_frames, ignore_index=True).to_csv(out_dir / "per_class_metrics.csv", index=False)
        metadata["failures"] = [list(f) for f in failures]
        _json_safe(metadata, out_dir / "run_metadata.json")
        return df

    for fold_idx, (train_index, test_index) in enumerate(splits, 1):
        log_message(f"\n{'='*50}")
        log_message(f"FOLD {fold_idx}/{k_external}")
        log_message(f"{'='*50}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        log_message(f"Fold {fold_idx} - Train: {len(X_train)}, Test: {len(X_test)}")
        metadata["folds"][str(fold_idx)] = {"n_train": int(len(X_train)), "n_test": int(len(X_test)), "models": {}}

        for model_name, model in MODELS.items():
            log_message(f"\n--- Testing model: {model_name} ---")
            info = metadata["folds"][str(fold_idx)]["models"][model_name] = {}
            try:
                t0 = time.time()
                row, art = test_classifier(X_train, y_train, X_test, y_test, model=model, classes=classes,
                                           k_internal=k_internal, scoring=scoring, scale_data=scale_data, n_jobs=n_jobs)
                elapsed = time.time() - t0
                row.update({"cls_name": model_name, "fold": fold_idx, "status": "OK", "error": "",
                            "time_seconds": elapsed})     # total time of this evaluation (search+fit+both predictions)
                rows.append(row)

                np.savez_compressed(out_dir / f"predictions_{model_name}_fold{fold_idx}.npz",
                                    test_index=test_index, y_true=y_test, y_pred=art["y_pred_test"])
                pd.DataFrame(art["confusion_matrix"], index=[f"true_{c}" for c in classes],
                             columns=[f"pred_{c}" for c in classes]).to_csv(out_dir / f"confusion_matrix_{model_name}_fold{fold_idx}.csv")
                pc = art["per_class"].copy(); pc.insert(0, "fold", fold_idx); pc.insert(0, "model", model_name)
                per_class_frames.append(pc)
                info.update({"status": "OK", "best_params": art["best_params"], "config": art["config"],
                             "epochs_completed": row["epochs_completed"], "time_seconds": elapsed})
                if art["history"] is not None:
                    hist = pd.DataFrame(art["history"], columns=["epoch", "loss", "monitor_accuracy"])
                    hist.to_csv(out_dir / f"training_history_{model_name}_fold{fold_idx}.csv", index=False)
                    info["training_history"] = art["history"]
                if art["initial_weights"] is not None:
                    info["initial_weights"] = np.asarray(art["initial_weights"]).tolist()
                    info["final_weights"] = np.asarray(art["final_weights"]).tolist()
                    qnn_initial[fold_idx] = np.asarray(art["initial_weights"])
                log_message(f"✓ {model_name} completed in {elapsed:.2f} seconds")
            except Exception as e:
                tb = traceback.format_exc()
                log_message(f"✗ FAILED: {model_name} in fold {fold_idx}: {type(e).__name__}: {e}")
                log_message(tb)
                failures.append((model_name, fold_idx, f"{type(e).__name__}: {e}"))
                rows.append({"cls_name": model_name, "fold": fold_idx, "status": "FAILED",
                             "error": f"{type(e).__name__}: {e}"})
                info.update({"status": "FAILED", "error": f"{type(e).__name__}: {e}"})

        process = psutil.Process(os.getpid())
        log_message(f"Memory usage after fold {fold_idx}: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        write_results(final=False)   # checkpoint after every fold

    if "penny_qnn" in MODELS:
        if not verify_qnn_initialization(qnn_initial, metadata["seeds"]["qnn_random_state"]):
            failures.append(("penny_qnn", "init-check", "initial weights differ from the seeded initialisation"))

    df_results = write_results(final=True)
    log_message(f"\nResults saved to: {out_dir}/")

    ok = df_results[df_results["status"] == "OK"]
    cols = ["train_score", "test_score", "inner_cv_score", "balanced_accuracy", "macro_f1", "weighted_f1",
            "macro_precision", "macro_recall", "search_time", "fit_time", "predict_train_time",
            "predict_test_time", "time_seconds"]
    log_message(f"\n{'='*50}")
    log_message(f"FINAL RESULTS SUMMARY (successful evaluations only: {len(ok)} of {len(df_results)}):")
    log_message(f"{'='*50}")
    if ok.empty:
        log_message("No evaluation succeeded: there is nothing to summarise.")
    else:
        summary = ok.groupby("cls_name")[[c for c in cols if c in ok.columns]].agg(["mean", "std"])
        summary.to_csv(out_dir / "summary_by_model.csv")
        with pd.option_context("display.width", 250, "display.max_columns", None):
            log_message(summary.round(4).to_string())
    return df_results, failures


# =============================================================================
# MAIN: LOAD DATA & RUN
# =============================================================================

if __name__ == "__main__":
    exit_code = 0
    try:
        log_message("=" * 50)
        log_message("STARTING EXPERIMENT V4" + (" (QUICK MODE)" if QUICK else ""))
        log_message("=" * 50)

        log_message("Loading data...")
        data = np.load("data/sentinel.npz")
        log_message(f"Data keys: {list(data.keys())}")
        log_message(f"BANDS shape: {data['bands'].shape}")
        log_message(f"CLASSES shape: {data['classes'].shape}")

        X = data["bands"].reshape(-1, data["bands"].shape[-1])
        y = data["classes"].ravel()
        log_message(f"Flattened data: {X.shape[0]} samples, {X.shape[1]} features")
        u, c = np.unique(y, return_counts=True)
        log_message(f"Class distribution: {dict(zip(u.tolist(), c.tolist()))}")

        if QUICK:
            X_sample, y_sample = stratified_subsample(X, y, n=3000, min_per_class=6, seed=0)
            qnn = MODELS["penny_qnn"]["estimator"]
            qnn.epochs, qnn.monitor_size = 2, 300
            log_message(f"QUICK MODE: {len(X_sample)} stratified pixels, QNN with {qnn.epochs} epochs")
        else:
            X_sample, y_sample = X, y
            log_message(f"Using FULL dataset: {len(X_sample)} samples")

        log_message("\n" + "=" * 50)
        log_message("RUNNING EXPERIMENT")
        log_message("=" * 50)
        start_time = time.time()

        results, failures = perform_experiment(
            X=X_sample, y=y_sample, instance_index=42, name="qnn_full_experiment",
            k_external=3, k_internal=2, scoring="accuracy", path=RESULTS_PATH, scale_data=True, n_jobs=1)

        total_time = time.time() - start_time
        log_message(f"\nTotal experiment time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")
        log_message("=" * 50)
        if failures:
            log_message(f"EXPERIMENT FINISHED WITH {len(failures)} FAILURE(S) - THIS IS NOT A SUCCESSFUL RUN:")
            for f in failures:
                log_message(f"  - {f}")
            exit_code = 1
        else:
            log_message("EXPERIMENT COMPLETED SUCCESSFULLY (all models, all folds, init check passed)")
        log_message("=" * 50)

    except Exception as e:
        log_message(f"FATAL ERROR: {str(e)}")
        log_message(traceback.format_exc())
        exit_code = 1

    sys.exit(exit_code)
