# DS605 Lab Assignment 7: PCA, Eigenfaces and Face Recognition

**Course:** DS605 — Fundamentals of Machine Learning

## Objective

Represent face images numerically, reduce image dimensionality using Principal Component Analysis (PCA), visualize eigenfaces, reconstruct images, perform face recognition, and compare PCA-based matching with traditional machine-learning classifiers.

## Dataset

This project uses the AT&T Database of Faces (ORL), containing 400 grayscale face images from 40 subjects, with 10 images per subject. Each image has dimensions of 112 × 92 pixels, giving 10,304 pixel features per image.

Download the dataset from the [AT&T Database of Faces archive](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

Extract the subject folders `s1` through `s40` into a local `data/` directory. The dataset is intentionally excluded from version control.

## Project Structure

```text
202618032_Ds605_lab7/
├── 202618032_DS605_Lab_7.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── figures/
└── results/
```

## Methodology

### Part A — Image Representation
- Load images as grayscale matrices and inspect their dimensions and pixel intensities.
- Flatten images into feature vectors and create the complete feature matrix and subject labels.
- Use a reproducible, subject-wise train-test split with random seed 42.
- Use seven training images and three test images for each subject.

### Part B — PCA, Eigenfaces and Compression
- Compute the mean face from training images and center the data.
- Implement PCA using NumPy eigendecomposition and Singular Value Decomposition (SVD).
- Compare the principal directions and eigenvalues from the two methods.
- Plot cumulative explained variance and select the minimum number of components retaining at least 90% variance.
- Visualize eigenfaces, project images into PCA space, and reconstruct sample faces.
- Compare reconstruction quality using MSE and PSNR and report storage requirements.

**Important:** PCA is implemented with NumPy linear-algebra operations. Scikit-learn PCA is not used.

### Part C — Face Recognition
- Use Euclidean distance to find the nearest training face for every test image.
- Compare raw-pixel matching with PCA-space matching.
- Report accuracy and inspect correct and incorrect matches.

### Part D — Traditional Machine Learning
- Train KNN, Logistic Regression, and SVM classifiers using NumPy-generated PCA features.
- Test multiple component counts and compare recognition accuracy.
- Save the model comparison table and plot.

## Results

Results below are from the current reproducible train-test split.

### PCA and Reconstruction

| Measurement | Result |
|---|---:|
| Training images | 280 |
| Test images | 120 |
| Original features per image | 10,304 |
| Components retaining at least 90% variance | 92 |
| Variance retained | 90.0366% |
| Test reconstruction MSE | 335.5241 |
| Test reconstruction PSNR | 22.87 dB |
| Per-image feature payload reduction | 92.86% |

The PCA coefficients are compact, but the full PCA basis and mean face must also be stored for reconstruction. For this small dataset, storing the complete PCA representation and basis uses more memory than storing all original uint8 images.

### Recognition Accuracy

| Method | Components | Accuracy |
|---|---:|---:|
| Raw-pixel 1-NN | 10,304 | 97.50% |
| PCA 1-NN | 92 | 97.50% |
| Logistic Regression | 92 | 95.00% |
| SVM (RBF) | 92 | 93.33% |
| KNN (k=3) | 92 | 75.00% |
| PCA 1-NN | 120 | 98.33% |

The highest observed accuracy was 98.33% using PCA nearest-neighbor matching with 120 components. These are results from one holdout split and are not a guarantee of performance on other datasets.

## Generated Figures

Running the notebook produces the following figures:

- `figures/pixel_intensity_distribution.png`
- `figures/mean_face.png`
- `figures/explained_variance.png`
- `figures/eigenfaces.png`
- `figures/reconstruction.png`
- `figures/recognition_matches.png`
- `figures/model_comparison.png`

Results, including recognition predictions and model comparison tables, are saved in `results/`.

## Installation and Execution

Python 3.14.6 was used for development.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` Python interpreter and use it as the Jupyter notebook kernel. Place the extracted ORL subject folders inside `data/`, open `202618032_DS605_Lab_7.ipynb`, and run all cells from top to bottom.

## Conclusion

PCA reduced the image representation from 10,304 pixel features to 92 components while retaining 90.0366% of the training variance. With 92 components, PCA nearest-neighbor matching achieved 97.50% test accuracy, matching the raw-pixel baseline in this experiment.

The results illustrate the trade-offs between dimensionality reduction, recognition performance, reconstruction quality, and total storage requirements. Traditional classifier performance varied with component count and model choice.

**Submission:** Public GitHub repository containing the runnable notebook, figures, results, requirements, and this README.
