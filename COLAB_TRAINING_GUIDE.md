# Google Colab GPU Acceleration Guide (Optional Speedup)

This guide is provided per project specification in case you wish to train with a high-end cloud GPU (such as NVIDIA T4 / A100 on Google Colab) instead of local CPU.

> **CRITICAL PRIVACY DIRECTIVE**:
> Never upload personal webcam clips (`New folder/`) to Google Colab or any external cloud service.
> Only upload `DATASET/`, `models/`, and `train_rafdb.py`.

---

## Step-by-Step Colab Instructions

### 1. Prepare Archive Locally (PowerShell)
Run the following PowerShell command in the project root to create a zip containing only the dataset and code:
```powershell
Compress-Archive -Path DATASET, models, train_rafdb.py -DestinationPath colab_training_bundle.zip
```

### 2. Open Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. Select **Runtime > Change runtime type** -> choose **T4 GPU** (or A100 if Colab Pro).

### 3. Upload and Extract
In a Colab notebook cell:
```python
# Upload the zip bundle
from google.colab import files
uploaded = files.upload() # Select colab_training_bundle.zip

# Unzip
!unzip -q colab_training_bundle.zip
!pip install timm hsemotion
```

### 4. Run Training with GPU (AMP & CUDA)
In Colab:
```bash
# Experiment A (Initialize from MELD production model)
python train_rafdb.py --experiment A --epochs 15 --stage1_epochs 3 --batch_size 64 --samples_per_class 600 --device cuda --output models/video_classifier/best_webcam_model_rafdb_A.pth --log_csv results/train_expA_log.csv

# Experiment B (Initialize from HSEmotion AffectNet backbone)
python train_rafdb.py --experiment B --epochs 15 --stage1_epochs 3 --batch_size 64 --samples_per_class 600 --device cuda --output models/video_classifier/best_webcam_model_rafdb_B.pth --log_csv results/train_expB_log.csv
```

### 5. Download Finished Checkpoints
```python
from google.colab import files
files.download('models/video_classifier/best_webcam_model_rafdb_A.pth')
files.download('models/video_classifier/best_webcam_model_rafdb_B.pth')
files.download('results/train_expA_log.csv')
files.download('results/train_expB_log.csv')
```
Place the downloaded files into their respective local directories (`models/video_classifier/` and `results/`).
