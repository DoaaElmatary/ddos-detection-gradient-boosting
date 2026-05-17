# ddos-detection-gradient-boosting

Balanced DDoS detection dataset (7,000 samples, 7 classes) with Gradient Boosting model achieving 95.90% accuracy.

##  Dataset

The dataset is **not included** in this repository due to file size limitations. 

Please download from **Mendeley Data** before running the code:

- **DOI:** 10.17632/55prxz2679.1
- **URL:** https://data.mendeley.com/datasets/55prxz2679.1

After downloading, place the file `ddos_balanced_7000_clean.csv` in the `data/` folder.

##  Requirements

- Python 3.8+
- 8 GB RAM

##  Steps to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/DoaaElmatary/ddos-detection-gradient-boosting.git
cd ddos-detection-gradient-boosting
2. Install dependencies
bash
pip install -r requirements.txt
3. Place dataset
Copy ddos_balanced_7000_clean.csv to data/ folder

4. Train models
bash
python src/02_train.py
Expected output:

Model	Accuracy	F1-Score
Gradient Boosting	95.90%	0.959
Random Forest	95.76%	0.958
Logistic Regression	91.43%	0.910
SVM (RBF)	91.38%	0.910
5. Generate figures (optional)
bash
python src/03_evaluate.py
Expected runtime: ~3 minutes

Troubleshooting
FileNotFoundError: Ensure CSV is in data/ folder

ModuleNotFoundError: Run pip install -r requirements.txt

Citation
bibtex
@data{elmatary_2026_ddos,
  author = {Elmatary, Doaa},
  title = {Dataset and Code for DDoS Detection using Gradient Boosting},
  publisher = {Mendeley Data},
  year = {2026},
  doi = {10.17632/55prxz2679.1}
}
License
CC BY 4.0
