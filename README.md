# landcover-sustainability
## Stanford CS230 Project 

#### Temporal analysis of regional sustainability using CNNs and satellite data. 
#### An investigation into landcover classification with Sentinel2 data.

### build_dataset.py
Takes RGB or 13 spectral band data and shuffles, splits into training/dev/test 80/20/20

### final_model.py
Trains final model using Resnet50 architecture. Computes validation and test set metrics, and exports model.

### export_satellite_data.ipynb
Notebook for exporting Sentinel2 data using the Google Earth Engine API. Data is exported as a tf record to a Google Cloud Storage bucket.

### make_predictions.py
Loads Google Earth Engine satellite data from Google Cloud Storage bucket as tf records. Parses tf records and computes predictions using the saved model.
