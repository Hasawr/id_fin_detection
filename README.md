# Azerbaijani ID Card FIN Code Detection & Extraction System

An automated OCR-based detection and extraction pipeline for Personal Identification Numbers (FIN / JŞV) from Azerbaijani National ID cards and Passports using **PaddleOCR** and **OpenCV**.

---

##  Features

- **Dual Extraction Engine**:
  - **VIZ (Visual Inspection Zone / Front Side)**: Detects FIN codes located under personal data fields using ROI extraction and pattern matching.
  - **MRZ (Machine Readable Zone / Back Side)**: Extracts 3-line document MRZ data, validates document checksums, and parses the 7-character FIN code.
- **Streamlit Web Application (`app.py`)**: Interactive UI for single-sided or dual-sided ID uploads with real-time confidence scores and visualization.
- **Command Line Interface (`main.py`)**: Scriptable CLI supporting JSON and formatted text outputs with optional debug visualization saving.
- **Checksum Verification**: Built-in verification algorithm for ICAO Doc 9303 MRZ format compliance.

---

##  Project Structure

```text
id_fin_detection/
├── fin_detector/          # Core detection module
│   ├── detector.py        # Main orchestrator (FINDetector)
│   ├── mrz_extractor.py   # MRZ side parser & checksum validator
│   ├── viz_extractor.py   # VIZ side ROI detection & regex parser
│   ├── preprocessor.py    # Image enhancement & grayscale transformations
│   └── validator.py       # Azerbaijani FIN validation logic
├── utils/
│   └── image_utils.py     # OpenCV drawing & visualization helpers
├── app.py                 # Streamlit web user interface
├── main.py                # Command Line Interface (CLI)
├── test_logic.py          # Unit & logic test suite
├── requirements.txt       # Python dependency file
└── .gitignore             # Git ignore file
```

---

##  Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/id_fin_detection.git
   cd id_fin_detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### 1. Streamlit Web App
Launch the interactive web application in your browser:
```bash
streamlit run app.py
```

### 2. Command Line Interface (CLI)
Run detection directly from the command line:

- **Both Sides (Front & Back)**:
  ```bash
  python main.py --viz images/front.png --mrz images/back.png --output json
  ```

- **Front Side Only**:
  ```bash
  python main.py --viz images/front.png --debug
  ```

- **Back Side Only**:
  ```bash
  python main.py --mrz images/back.png
  ```

---

##  Recommended Best Practices & Privacy

- **Sensitive Data**: Avoid committing real ID cards or personal documents containing PII (Personally Identifiable Information) to public repositories.
- **Debug Files**: Output files in `debug_output/` and `documents/` are ignored by `.gitignore` by default.
- **Git Ignore**: Virtual environments (`venv/`), IDE config files (`.idea/`, `.vscode/`), and temporary images are untracked.

---

## 📄 License

Distributed under the MIT License.
