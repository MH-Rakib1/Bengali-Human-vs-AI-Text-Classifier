# Bengali-Human-vs-AI-Text-Classifier

This project aims to classify Bengali text as either human-written or LLM-generated using machine learning techniques, such as Word2Vec for document embedding and Integrated Syntactic Graph (ISG) for extracting syntactic features. The final classification is performed using a Random Forest classifier.

## Requirements
Before running this code, ensure you have the following Python packages installed:

    spacy (for text processing)
    numpy (for numerical operations)
    gensim (for Word2Vec)
    networkx (for graph operations)
    sklearn (for machine learning models)
    bangla_stemmer (for stemming Bengali words)
    matplotlib (for plotting the confusion matrix)

You can install the required packages using pip: pip install spacy numpy gensim networkx scikit-learn matplotlib bangla-stemmer
Also, you need to download the Bengali language model for spaCy

Project Structure

.
├── dataset/
│   ├── bn_human_vs_ai_corpus/
│   │   ├── human_written/
│   │   └── llm_generated/
├── main.py
└── README.md

    dataset/bn_human_vs_ai_corpus/: Contains two folders:
        human_written/: Text files containing human-written Bengali content.
        llm_generated/: Text files containing LLM-generated Bengali content.

How It Works
1. Preprocessing Text

    The text is preprocessed by:
        Removing punctuation, digits, and non-Bengali characters.
        Tokenizing the text and applying stemming.
        Removing Bengali stop words.

2. Word2Vec Embedding

    A Word2Vec model is trained on the human-written and LLM-generated texts to create word embeddings for each token.

3. Integrated Syntactic Graph (ISG)

    For each document, a syntactic graph is built using spaCy, where nodes represent words and edges represent syntactic dependencies between words.

4. Feature Extraction

    Features are extracted by combining Word2Vec embeddings and graph-based features, such as:
        Degree sequence (average, max, min degree).
        Graph properties (density, radius, diameter, etc.).
        Chromatic number and clustering coefficient.

5. Model Training

    A Random Forest classifier is trained on the extracted features to distinguish between human-written and LLM-generated text.

6. Evaluation

    The classifier is evaluated using metrics such as accuracy, precision, recall, and F1-score.
    A confusion matrix is plotted to visualize the model's performance.

7. Classifying New Text

    A new document can be classified as human-written or LLM-generated using the trained model.

How to Run

    Prepare the dataset: Place your text files into the respective directories human_written and llm_generated.
    Train the model: The script will automatically load and preprocess the texts, train the Word2Vec models, extract features, and train the classifier.
    Classify a new document: To classify a new text file, use the classify_single_document function with the path to your file.

Example Usage:

# Define the file path to the txt file
file_path = "dataset/bn_human_vs_ai_corpus/llm_generated/shapure_choto_golpo_ai.txt"  # Replace with the actual file path

# Call the function to classify the document
classify_single_document(file_path, clf, human_w2v_model, llm_w2v_model)

This will output the classification result as either Human-written or LLM-generated.
Evaluation

After training, the model's performance is evaluated using the following metrics:

    Accuracy
    Precision
    Recall
    F1 Score

Additionally, a confusion matrix is plotted to visualize the classification performance.
