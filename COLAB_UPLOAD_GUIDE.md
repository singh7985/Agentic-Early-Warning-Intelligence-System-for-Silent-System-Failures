# 📤 Upload Project to Google Colab GPU

## Quick Start (5 minutes)

You're using VS Code with Colab GPU extension. Your local files need to be uploaded to Colab's cloud.

### Step 1: Create ZIP File (30 seconds)

Open Terminal on your Mac and run:

```bash
cd "/Users/xe/Documents/GITHUB CAPSTONE /Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures"
./prepare_for_colab.sh
```

This creates `colab_project.zip` on your Desktop (~50-200 MB).

**What's included:**
- ✅ `src/` - Your Python code
- ✅ `data/processed/` - Your preprocessed datasets
- ✅ `configs/` - Configuration files
- ✅ `models/checkpoints/` - Model save directory
- ✅ `requirements.txt` - Dependencies
- ❌ Excludes: `.git`, `.venv`, `__pycache__`, raw data, logs (saves space & time)

### Step 2: Upload to Colab (2-5 minutes)

1. **Open your notebook in VS Code** (already connected to Colab)

2. **Find the Colab file browser:**
   - Look for the 📁 folder icon on the left sidebar
   - You should see `/content/` directory

3. **Upload the ZIP:**
   - Click the upload button (⬆️ icon)
   - Select `colab_project.zip` from your Desktop
   - Wait for upload (2-5 min depending on file size and internet speed)

### Step 3: Extract Files (30 seconds)

In your notebook, there's a cell titled **"QUICK UPLOAD COMMANDS"**. Run that cell to:
- Extract the ZIP file
- Verify all folders are present
- Check your datasets are there

You should see:
```
✅ Project folder found
✅ src/
✅ data/processed/
✅ models/
✅ configs/
📊 Found 3 CSV files in data/processed/
  - train_features.csv (XX MB)
  - val_features.csv (XX MB)
  - test_features.csv (XX MB)
🎉 All files uploaded successfully!
```

### Step 4: Start Training! 🚀

Once verified, run the remaining cells in order:
1. **Section 1:** Setup & Imports (imports your code from `src/`)
2. **Section 2:** Load Data (loads your uploaded datasets)
3. **Section 3-6:** Train models (XGBoost, RF, GB, LSTM, TCN) on Colab GPU
4. **Section 7-8:** Compare models and save results

---

## Troubleshooting

### "Project folder not found"
- Make sure you uploaded `colab_project.zip` to `/content/` (not inside a subfolder)
- Re-run the extraction cell

### "src/ directory not found"
- The ZIP extraction may have failed
- Try uploading the ZIP file again
- Make sure the ZIP was created by the `prepare_for_colab.sh` script

### "Data files not found"
- Check that your `data/processed/` folder has the CSV files locally
- If missing, run the preprocessing notebook first (02_feature_engineering_pipeline.ipynb)
- Re-create the ZIP after preprocessing

### Upload is taking too long
- Check your internet upload speed
- The ZIP should be 50-200 MB
- If larger, the script might have included unnecessary files
- You can check ZIP contents: `unzip -l ~/Desktop/colab_project.zip`

### Still having issues?
Alternative: **Clone from GitHub**

If your project is on GitHub:
```python
# In a Colab cell:
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures
```

---

## What Data Gets Uploaded?

The script includes:
- **Processed features:** `data/processed/train_features.csv`, `val_features.csv`, `test_features.csv`
- **Your source code:** Everything in `src/` directory
- **Configuration:** `configs/` folder
- **Space for models:** `models/checkpoints/` (empty, for saving trained models)

**NOT included** (to save space):
- Raw C-MAPSS data (not needed, already processed)
- Virtual environments
- Git history
- Log files
- Cache files

Total size should be manageable for quick uploads!

---

## After Training

Your trained models and results will be saved in:
- `/content/Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures/models/checkpoints/`
- `/content/Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures/reports/figures/`

**Download results:**
1. Right-click files in Colab file browser
2. Select "Download"
3. Save to your local Mac

Or use this command in a cell:
```python
# Compress results for download
!cd /content && zip -r results.zip Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures/models/checkpoints/ Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures/reports/
```

Then download `results.zip` from the file browser.

---

**🎉 You're all set! Enjoy training on Colab GPU!**
