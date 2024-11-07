# Emotion Classification with Naive Bayes and BERT Models

## Overview
This project explores the application of machine learning and deep learning techniques to classify emotions in text data. It evaluates the performance of a traditional Multinomial Naive Bayes model and two BERT-based deep learning approaches (pre-trained and fine-tuned) on a labeled dataset of tweets. The results highlight the superior performance of BERT models in capturing the nuances of emotional expressions in text.

## Features
- **Dataset Preprocessing**: Tweets are preprocessed to remove noise (usernames, URLs, and extra spaces) and tokenized for model training.
- **Multinomial Naive Bayes Model**: Utilizes word frequency features to classify emotions with competitive performance.
- **Pre-trained BERT Model**: Implements a pre-trained `bert-base-uncased-emotion` model for emotion classification with minimal additional training.
- **Fine-tuned BERT Model**: Fine-tunes the pre-trained BERT model on the dataset for improved accuracy.
- **Performance Analysis**: Detailed comparison of model accuracy and analysis of classification results using attention matrices.

## Dataset
The dataset originates from the CARER paper, where tweets are labeled with one of six emotions: *Sadness*, *Joy*, *Fear*, *Anger*, *Surprise*, and *Love*. Preprocessing steps include tokenization and removal of irrelevant features, with further transformations for input to the models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-classification.git
   cd emotion-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Models
1. **Naive Bayes Model**:
   ```bash
   python naive_bayes.py
   ```
2. **Pre-trained BERT Model**:
   ```bash
   python pretrained_bert.py
   ```
3. **Fine-tuned BERT Model**:
   ```bash
   python fine_tuned_bert.py
   ```

### Notebook
Explore the full workflow and experiments interactively using the provided Jupyter Notebook:
```bash
jupyter notebook COMP551_Project3.ipynb
```

## Results
- **Multinomial Naive Bayes**: 86% accuracy
- **Pre-trained BERT**: 92% accuracy
- **Fine-tuned BERT**: 93.9% accuracy

### Performance Insights
- Fine-tuning BERT provides a significant improvement in accuracy by adjusting model-specific parameters such as learning rate and batch size.
- The Naive Bayes model, while efficient, struggles with feature independence assumptions in emotion classification tasks.

## Contributing
Contributions are welcome! If you have suggestions for improving the code or extending the project, please:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## Contributors
This project was developed as part of COMP551 by the following contributors:

Edwin Zhou (260988798)
Harry MacFarlane (260991258)
Yongru Pan (261001758)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **CARER Paper**: Data preprocessing and methodology inspiration from the work of Elvis Saravia et al. (2018).
- **Transformers Library**: For seamless BERT model implementation.
- **Scikit-learn**: For Naive Bayes and feature engineering utilities.
