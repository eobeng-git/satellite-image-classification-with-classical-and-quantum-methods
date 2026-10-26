# Satellite image classification with classical and quantum methods

Master's thesis project (University of L'Aquila, DISIM). Benchmarks SVM, KNN, Decision Tree
and Naive Bayes against a 4-qubit PennyLane quantum neural network (PennyQNN) on Sentinel-2
land-cover pixels (512x512 pixels, 12 bands, 10 classes) with 3-fold cross-validation.

## Mean test accuracy
KNN 81.52% | SVM 80.19% | Decision Tree 76.00% | PennyQNN 68.61% | Naive Bayes 66.36%

## Files
- alg_experiment_qnn_fixed_V2_multiclass.py : experiment (classical models + PennyQNN)
- run_qnn_full_multiclass.sh : SLURM job script (edit the `cd` line for your path)
- make_thesis_figures.py : regenerates the thesis figures from the results
- results_qnn_full/ and job-progress.log : results and training log of the final run
- environment.yml : conda environment

## Setup and run
    conda env create -f environment.yml -n qnn-310
    conda activate qnn-310
    sbatch run_qnn_full_multiclass.sh
    python make_thesis_figures.py

The dataset (data/sentinel.npz) is not included.
