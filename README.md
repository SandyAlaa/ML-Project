Email/SMS Spam Classifier

Overview
This project is a **machine learning-based spam classifier** that predicts whether an email or SMS message is spam or not. It utilizes NLP techniques to process text data and train a model for classification.

Dataset
The dataset used for training consists of labeled messages categorized as spam or ham (not spam). It was preprocessed to remove noise and tokenize the text before feeding it into the model.

Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Pickle (for model saving & loading)
- Jupyter Notebook
  
Model Training Process
- Text Processing: Tokenization, stopword removal, and vectorization using CountVectorizer/TfidfVectorizer.
- Training: Used a classification model such as Logistic Regression or Random Forest.
- Evaluation: Measured accuracy, precision, recall, and F1-score to validate performance.
- Saving the Model: Stored using Pickle for easy reuse.

How to Run the Project
1. Clone this repository:
   git clone https://github.com/SandyAlaa/ML-Project.git
   cd Email-SMS-Spam-Classifier
 
2. Install dependencies:
   pip install -r requirements.txt
  
3. Run the Jupyter Notebook:
   jupyter notebook Email_SMS_Spam_Classifier.ipynb
  
4. Alternatively, use the pre-trained model for prediction:
   import pickle
   Load vectorizer and model
   with open("vectorizer.pkl", "rb") as v_file, open("model.pkl", "rb") as m_file:
       vectorizer = pickle.load(v_file)
       model = pickle.load(m_file)
   Example message
   sample_text = ["Congratulations! You've won a free ticket to Bahamas!"]
   transformed_text = vectorizer.transform(sample_text)
   prediction = model.predict(transformed_text)
   print("Spam" if prediction[0] == 1 else "Not Spam")

Usage Example
Provide a message, and the classifier will predict if it's spam or not.

Input: "You have won a free lottery! Click here to claim."
Output: Spam

Input: "Hey, are we still meeting for lunch today?"
Output: Not Spam
