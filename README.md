# Hindi to Indian Sign Language Translation App

## Overview
This application translates Hindi text or speech into Indian Sign Language (ISL), making communication more accessible between Hindi speakers and the deaf community.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required language models:
```bash
python -c "import stanza; stanza.download('hi')"
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Features
- Hindi text to ISL translation
- Hindi speech to ISL translation
- Real-time translation capabilities
- User-friendly interface
- Dependency parsing visualization
- Fingerspelling support

## Configuration
Set your ISL videos directory path in the application:
```python
folder_path = "Your ISL Videos Path"
```

## Usage
To run the application:
```bash
streamlit run app.py
```

## Required Directory Structure
```
Hindi_To_ISL_App/
│
├── app.py
├── translator.py
├── requirements.txt
├── README.md
│
└── utility/
    ├── isl_dict.txt
    ├── final_stopwords.txt
    └── output.txt
```
