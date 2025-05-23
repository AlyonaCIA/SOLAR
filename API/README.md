
# Anomaly Detection and Clustering Pipeline

This pipeline performs anomaly detection on images (in FITS or JP2 format) and applies clustering using the K-Means algorithm. The main steps are:

1. **Loading and Preprocessing Data**: Loads solar images, applies masks, and prepares the data for anomaly detection.
2. **Anomaly Detection**: Uses the Isolation Forest algorithm to identify anomalies in the images.
3. **Thresholding and Clustering**: Applies multiple thresholds to classify anomalies, followed by clustering with K-Means.
4. **Visualization**: Generates visualizations such as anomaly maps and clustering results.

---

## Requirements

This pipeline requires the following libraries:

- `numpy`
- `matplotlib`
- `tqdm`
- `scikit-learn`
- `pandas`

You can install these dependencies using `pip` or a `requirements.txt` file:

```bash
pip install numpy matplotlib tqdm scikit-learn pandas
```

Ensure you activate your virtual environment before installing the dependencies if you're using one.

---

## Project Structure

The project directory is organized as follows:

```
├── data_loader.py           # Functions for loading image data
├── model.py                 # Functions for anomaly detection and clustering
├── preprocess.py            # Functions for preprocessing the input data
├── visualization.py         # Functions for generating plots and visual outputs
├── pipeline.py              # Main script that runs the full pipeline
├── output_figures/          # Directory for saving generated visualizations
├── config.py                # Configuration file for pipeline parameters
```

---

## Input Files

The pipeline expects input images in **FITS** or **JP2** format, depending on the `file_type` parameter in the configuration. Input images should match the specified size (default: **2048x2048** pixels) and be located in the folder defined by `data_dir` in the configuration file.

---

## Usage

### 1. Configuration

Edit the `config.py` file to set the desired pipeline parameters. Key configuration options include:

- `data_dir`: Directory containing the input image files.
- `output_dir`: Directory where output visualizations will be saved.
- `image_size`: Size of input images (default: 2048).
- `anomaly_thresholds`: List of thresholds to apply during anomaly detection.
- `n_clusters`: Number of clusters to use in K-Means clustering.

---

### 2. Running the Pipeline

Once configured, run the pipeline with:

```bash
python pipeline.py
```

The pipeline executes the following:

1. **Output directory setup**: Creates the output directory if it does not exist.
2. **Data loading and preprocessing**: Reads the images and prepares them for analysis.
3. **Anomaly detection**: Applies Isolation Forest to identify anomalous regions.
4. **Clustering**: Uses K-Means to group the detected anomalies.
5. **Visualization**: Saves generated plots (anomaly and clustering maps) to the output directory.

---

### 3. Output

After execution, the pipeline will save visual results in the `output_dir` folder, including:

- Anomaly detection maps.
- Clustering maps generated using K-Means.

---

### 4. Customization

You can easily customize the behavior of the pipeline by modifying the `config.py` file. Adjustable parameters include:

- Anomaly detection thresholds.
- Number of clusters for K-Means.
- Hyperparameters for the Isolation Forest and K-Means algorithms.

---

## Contributing

Contributions are welcome! You can:

- Open issues to report bugs or suggest enhancements.
- Submit pull requests with new features, improvements, or fixes.
