import pandas as pd
import numpy as np
import re
import html

# 1. Load dataset
df = pd.read_csv("fake_job_postings.csv")

print("Original Dataset Shape:", df.shape)

# 2. Remove unnecessary column
df.drop("job_id", axis=1, inplace=True)

# 3. Identify text columns
text_columns = [
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

# 4. Handle missing values
for col in text_columns:
    df[col] = df[col].fillna("")

# 5. Text cleaning function
def clean_text(text):

    text = html.unescape(str(text))

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# 6. Apply text cleaning
for col in text_columns:
    df[col] = df[col].apply(clean_text)


# 7. Combine text columns
df["combined_text"] = df[text_columns].agg(' '.join, axis=1)

# 8. Check missing values
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# 9. Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# 10. Display dataset shape
print("\nFinal Dataset Shape:", df.shape)

# 11. Save preprocessed dataset
df.to_csv("preprocessed_fake_jobs.csv", index=False)

print("\nPreprocessing Completed Successfully!")

# STEP 2: FEATURE EXTRACTION AND MODEL TRAINING

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib

# 1. Separate input and output variables

X = df["combined_text"]
y = df["fraudulent"]

# 2. Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))


# 3. Create TF-IDF and Logistic Regression pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])


# 4. Train the model

print("\nTraining the Machine Learning Model...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


# 5. Test the model

y_pred = model.predict(X_test)


# 6. Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print("Accuracy:", round(accuracy * 100, 2), "%")


# 7. Classification report

print("\nCLASSIFICATION REPORT:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine Job", "Fake Job"],
    zero_division=0
))


# 8. Confusion matrix

print("\nCONFUSION MATRIX:")

print(confusion_matrix(y_test, y_pred))


# 9. Save trained model

joblib.dump(model, "fake_job_model.pkl")

print("\nTrained Model Saved Successfully!")

# ==========================================
# STEP 3: GUIDED JOB VERIFICATION
# ==========================================

def get_input(question, example):
    print("\n" + question)
    print("Example:", example)
    return input("Enter your answer: ")


def get_choice(question, options):

    print("\n" + question)

    for key, value in options.items():
        print(key + ".", value)

    while True:
        choice = input("Select your option: ").strip().lower()

        if choice in options:
            return options[choice]

        print("Invalid choice! Please select again.")


def predict_job():

    print("\n===================================")
    print("       ENTER JOB DETAILS")
    print("===================================")

    title = get_input(
        "1. Job Title",
        "Python Developer Intern"
    )

    company_profile = get_input(
        "2. Company Profile",
        "Software development company"
    )

    description = get_input(
        "3. Job Description",
        "Developing Python applications"
    )

    salary_range = get_input(
        "4. Salary / Stipend",
        "10000-15000 per month"
    )

    requirements = get_input(
        "5. Required Skills",
        "Python, SQL, Communication"
    )

    benefits = get_input(
        "6. Benefits Offered",
        "Monthly stipend and certificate"
    )

    experience = get_choice(
        "7. Required Experience",
        {
            "a": "No Experience",
            "b": "Fresher",
            "c": "1-2 Years",
            "d": "3+ Years",
            "e": "Not Specified"
        }
    )

    education = get_choice(
        "8. Educational Qualification",
        {
            "a": "10th",
            "b": "12th",
            "c": "Diploma",
            "d": "Bachelor's Degree",
            "e": "Master's Degree",
            "f": "Any Qualification"
        }
    )

    employment = get_choice(
        "9. Employment Type",
        {
            "a": "Internship",
            "b": "Full Time",
            "c": "Part Time",
            "d": "Contract",
            "e": "Other"
        }
    )

    registration_fee = get_choice(
        "10. Is Registration Fee Required?",
        {
            "a": "Yes",
            "b": "No",
            "c": "Not Sure"
        }
    )

    interview = get_choice(
        "11. Is an Interview Conducted?",
        {
            "a": "Yes",
            "b": "No",
            "c": "Not Sure"
        }
    )

    # Combine user input into text
    job_text = " ".join([
        title,
        company_profile,
        description,
        salary_range,
        requirements,
        benefits,
        experience,
        education,
        employment,
        "Registration fee: " + registration_fee,
        "Interview: " + interview
    ])

    # Apply preprocessing
    job_text = clean_text(job_text)

    # Model prediction
    prediction = model.predict([job_text])[0]

    probabilities = model.predict_proba([job_text])[0]

    fake_probability = probabilities[1] * 100

    print("\n===================================")
    print("          ANALYSIS RESULT")
    print("===================================")

    if prediction == 1:
        print("STATUS: POTENTIALLY FAKE")
    else:
        print("STATUS: LIKELY GENUINE")

    print("Estimated Fraud Probability:",
          round(fake_probability, 2), "%")

    print("\nPlease verify the company independently.")
    print("===================================")


while True:

    choice = input(
        "\nDo you want to verify a job? (yes/no): "
    )

    if choice.lower() in ["yes", "y"]:
        predict_job()

    else:
        print("\nThank you for using AI Fake Job Detector!")
        break