# Meta-Data-Extraction-from-Documents
This project focuses on developing an AI/ML system for extracting metadata from documents regardless of their template format. Users can upload scanned images or .docx files, and the system extracts key fields such as Agreement Value, Agreement Start Date, Agreement End Date, Renewal Notice (Days), Party One, and Party Two. The system avoids rule-based approaches, ensuring flexibility across various document types.

## Problem Statement
This project aims to develop an AI/ML system capable of extracting metadata from documents, irrespective of their template format. Users can upload scanned images or .docx files, and the system extracts the following fields:
- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

The system avoids rule-based approaches (such as RegEx or static conditions) to ensure flexibility across various document types.

## Dataset Details
The dataset is organized in the following structure:
- `train.csv`: Contains metadata details of files under the `train/` folder
- `test.csv`: Contains metadata details of files under the `test/` folder
- `train/`: Contains .docx and .png files for training
- `test/`: Contains .docx and .png files for testing/evaluation

## Evaluation Criteria
The system's performance will be evaluated based on per-field recall, which refers to the number of exact value matches for a document's metadata given in the training/validation set compared to the extracted value by the system.

## Code
The code is written in Python and utilizes libraries such as Keras, TensorFlow, pandas, numpy, and EasyOCR. It includes:
- Data preprocessing
- Model training and evaluation
- Text extraction from document files (.docx) and images (.png) using EasyOCR

```
## Directory Structure
ML_Project/
├── data/
│ ├── train/
│ │ ├── docs/
│ │ └── images/
│ └── test/
│ ├── docs/
│ └── images/
├── notebooks/
│ └── your_notebook.ipynb
├── scripts/
│ └── your_script.py
├── README.md
└── .gitignore
```

## Usage
1. Clone the repository to your local machine.
2. Navigate to the root directory.
3. Install the required dependencies (`requirements.txt`).
4. Run the Jupyter Notebook [Meta Data Extraction from Documents.ipynb](notebooks/Meta%20Data%20Extraction%20from%20Documents.ipynb)
 or execute the Python script (`MetadataExtraction.py`).
5. Evaluate the model's performance using the provided dataset.

## License
[Creative Commons](LICENSE)

## Acknowledgements
- This project utilizes the EasyOCR library for text extraction from images.
- The machine learning models are built using TensorFlow and Keras.
- We acknowledge scikit-learn, Matplotlib, pandas and NumPy libraries for providing useful tools for model evaluation.
- The authors of the OpenAI GPT-3.5 model, which powers the language model providing assistance in this project.
- We extend our gratitude to the GitHub community for providing a platform for collaboration and version control.
- My thanks to all contributors, developers, and maintainers of open-source projects that have indirectly supported this work.

Feel free to contribute, report issues, or provide feedback!






