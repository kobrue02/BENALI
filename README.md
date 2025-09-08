# BENALI: BERT for NAtive Language Identification

This repository contains the implementation of BENALI, a multilingual native language identification system that predicts a writer's first language (L1) based on their writing in a second language (L2).

## Overview

BENALI tackles the challenging task of identifying the native language of writers based on linguistic patterns and interference phenomena that appear when writing in a second language. The project implements multiple approaches including traditional lexical features, transformer-based models, and hybrid CNN architectures.

## Dataset

The project uses a comprehensive multilingual dataset combining several sources:

- **English L2 texts**: 3 sources including native language recognition datasets and English learner corpora
- **Czech L2 texts**: CZESL corpus with various L1 backgrounds
- **Slovenian L2 texts**: KOST 2.0 learner corpus
- **Portuguese L2 texts**: CRPC corpus
- **Chinese L2 texts**: JCLC corpus

**Dataset Statistics:**
- Total samples: ~800K sentences after preprocessing
- Languages covered: 90+ native languages
- L2 target languages: English, Czech, Slovenian, Portuguese, Chinese
- Balanced dataset with upsampling for minority classes

## Models and Approaches

### 1. Traditional Baselines (`lexical_svm_baselines.ipynb`)
- **TF-IDF + SVM**: Character n-grams (3-5) with Linear SVM
- **Majority Baseline**: Always predicts most frequent class
- **Random Baseline**: Stratified random predictions
- Performance: ~28% accuracy on test set

### 2. Transformer Fine-tuning (`finetuning.ipynb`)
- **BERT Multilingual**: Fine-tuned bert-base-multilingual-cased
- Data augmentation with character-level noise, deletions
- Batch size: 64, Learning rate: 5e-5, Epochs: 5
- Performance: ~34% accuracy on test set

### 3. Deep Neural Networks (`classifier.ipynb`)
- **BERT + Linear Layers**: Deep multilingual classifier with BERT embeddings
- **CNN Architecture**: 1D CNN over BERT representations
- Multiple embedding models tested (BERT, XLM-RoBERTa, Sentence Transformers)

### 4. Intermediate CNN (`interCNN.ipynb`)
- **TransformerCNN**: Extracts intermediate transformer representations
- 1D CNN over concatenated hidden states from multiple layers
- Layer indices: [-7, -6, -5, -4] for capturing different linguistic levels

## Repository Structure

```
BENALI/
├── dataset_curation.ipynb          # Data collection and preprocessing
├── notebooks/
│   ├── classifier.ipynb            # Deep learning models
│   ├── classifier_traditional.ipynb
│   ├── finetuning.ipynb           # Transformer fine-tuning
│   ├── interCNN.ipynb             # Intermediate CNN approach
│   ├── lexical_svm_baselines.ipynb # Traditional ML baselines
│   └── preprocessing.ipynb         # Data preprocessing utilities
├── nli_train_upsampled.csv        # Balanced training data
├── nli_val.csv                    # Validation set
├── nli_test.csv                   # Test set
└── README.md
```

## Key Features

- **Multilingual Support**: Handles 5 different L2 languages
- **Data Augmentation**: Character-level noise injection for balancing
- **Multiple Architectures**: From traditional ML to state-of-the-art transformers
- **Comprehensive Evaluation**: Cross-validation and detailed classification reports
- **Experiment Tracking**: Integration with Weights & Biases (wandb)

## Installation

```bash
# Clone the repository
git clone https://github.com/kobrue02/BENALI.git
cd BENALI

# Install required packages
pip install pandas numpy scikit-learn
pip install torch torchvision transformers
pip install datasets evaluate accelerate
pip install wandb nlpaug sentence-transformers
pip install beautifulsoup4 lxml
```

## Usage

### 1. Data Preparation
```python
# Run dataset_curation.ipynb to collect and preprocess data
# This will create the train/val/test splits
```

### 2. Traditional Baselines
```python
# Run lexical_svm_baselines.ipynb for TF-IDF + SVM approach
```

### 3. Transformer Fine-tuning
```python
# Run finetuning.ipynb for BERT fine-tuning
```

### 4. Advanced Models
```python
# Run classifier.ipynb or interCNN.ipynb for deep learning approaches
```

## Performance

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|--------|
| Random Baseline | 3% | 1% | Stratified random |
| Majority Baseline | 8% | 15% | Always predicts Germany |
| TF-IDF + SVM | 28% | 14% | Character n-grams |
| BERT Fine-tuned | 34% | 18% | bert-base-multilingual-cased |
| TransformerCNN | - | - | Intermediate representations |

## Challenges Addressed

1. **Class Imbalance**: Addressed through upsampling and data augmentation
2. **Multilingual Complexity**: Different L2 languages require different approaches
3. **Linguistic Interference**: Models capture L1 transfer patterns in L2 writing
4. **Scalability**: Efficient processing of large multilingual datasets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work, please cite:
```bibtex
@misc{benali2024,
  title={BENALI: BERT for NAtive Language Identification},
  author={[Konrad Brüggemann]},
  year={2025},
  howpublished={GitHub Repository},
  url={[https://github.com/kobrue02/BENALI]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CZESL corpus for Czech L2 data
- KOST 2.0 for Slovenian L2 data
- CRPC for Portuguese L2 data
- JCLC for Chinese L2 data
- Various English L2 datasets from research communities
