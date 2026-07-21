from flask import Flask, request, render_template_string
import pickle
import numpy as np

# Load model
with open("rfmodel.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Prediction</title>
    <style>
        body{
            font-family:Arial;
            background:#f4f4f4;
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
        }
        .box{
            background:white;
            padding:30px;
            border-radius:10px;
            width:400px;
            box-shadow:0px 0px 10px gray;
        }
        input{
            width:100%;
            padding:10px;
            margin:8px 0;
        }
        button{
            width:100%;
            padding:10px;
            background:#28a745;
            color:white;
            border:none;
            cursor:pointer;
            font-size:16px;
        }
        h2{text-align:center;}
        h3{text-align:center;color:blue;}
    </style>
</head>
<body>
<div class="box">
<h2>Student Performance Prediction</h2>

<form method="POST">

<input type="number" step="any" name="weekly_self_study_hours"
placeholder="Weekly Self Study Hours" required>

<input type="number" step="any" name="attendance_percentage"
placeholder="Attendance Percentage" required>

<input type="number" step="any" name="class_participation"
placeholder="Class Participation" required>

<input type="number" step="any" name="total_score"
placeholder="Total Score" required>

<button type="submit">Predict</button>

</form>

{% if prediction %}
<h3>{{ prediction }}</h3>
{% endif %}

</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        features = [
            float(request.form["weekly_self_study_hours"]),
            float(request.form["attendance_percentage"]),
            float(request.form["class_participation"]),
            float(request.form["total_score"])
        ]

        pred = model.predict([features])[0]

        prediction = f"Prediction : {pred}"

    return render_template_string(HTML, prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
