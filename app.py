from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load trained machine learning model
MODEL_PATH = os.path.join("model", "fake_job_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/verify", methods=["GET", "POST"])
def verify():

    if request.method == "POST":

        title = request.form.get("title", "")
        company_profile = request.form.get("company_profile", "")
        description = request.form.get("description", "")
        requirements = request.form.get("requirements", "")
        benefits = request.form.get("benefits", "")
        location = request.form.get("location", "")
        salary_range = request.form.get("salary_range", "")
        employment_type = request.form.get("employment_type", "")
        experience = request.form.get("experience", "")
        education = request.form.get("education", "")

        # Combine details into one text input
        job_text = " ".join([
            title,
            location,
            salary_range,
            company_profile,
            description,
            requirements,
            benefits,
            employment_type,
            experience,
            education
        ])

        # Predict job authenticity
        prediction = model.predict([job_text])[0]

        # Get probability if supported
        probabilities = model.predict_proba([job_text])[0]
        fake_probability = probabilities[list(model.classes_).index(1)] * 100

        if prediction == 1:
            result = "Potentially Fake Job"
            risk = "High Risk"
        else:
            result = "Likely Genuine Job"
            risk = "Low Risk"

        return render_template(
            "result.html",
            result=result,
            risk=risk,
            probability=round(fake_probability, 2),
            title=title
        )

    return render_template("verify.html")


if __name__ == "__main__":
    app.run(debug=True)