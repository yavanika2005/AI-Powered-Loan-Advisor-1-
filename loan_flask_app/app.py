from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, g
import pickle
import numpy as np
import os
import sqlite3
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin

app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY="replace_this_with_a_random_secret key",
    DATABASE=os.path.join(app.instance_path, "users.db"),
)

# ensure instance folder exists
try:
    os.makedirs(app.instance_path, exist_ok=True)
except OSError:
    pass

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id_, email):
        self.id = id_
        self.email = email

    def get_id(self):
        return str(self.id)

# Database helpers
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(app.config["DATABASE"])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def query_user_by_email(email):
    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE email = ?", (email,))
    return cur.fetchone()

def query_user_by_id(id_):
    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE id = ?", (id_,))
    return cur.fetchone()

@login_manager.user_loader
def load_user(user_id):
    row = query_user_by_id(user_id)
    if row:
        return User(row["id"], row["email"])
    return None

# Load ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model = pickle.load(open('model.pkl', 'rb'))")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print("Warning: could not load model.pkl:", e)

def preprocess_data(gender, married, dependents, education, employed, credit, area,
                    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    male = 1 if str(gender).lower().startswith("m") else 0
    married_yes = 1 if str(married).lower().startswith("y") else 0
    if str(dependents) == '1':
        dependents_1, dependents_2, dependents_3 = 1, 0, 0
    elif str(dependents) == '2':
        dependents_1, dependents_2, dependents_3 = 0, 1, 0
    elif str(dependents) == '3+':
        dependents_1, dependents_2, dependents_3 = 0, 0, 1
    else:
        dependents_1, dependents_2, dependents_3 = 0, 0, 0

    not_graduate = 1 if str(education).lower().startswith("n") else 0
    employed_yes = 1 if str(employed).lower().startswith("y") else 0
    semiurban = 1 if str(area).lower().startswith("s") else 0
    urban = 1 if str(area).lower().startswith("u") else 0

    try:
        ApplicantIncomelog = np.log(float(ApplicantIncome) if float(ApplicantIncome) > 0 else 1.0)
    except:
        ApplicantIncomelog = 0.0
    try:
        totalincomelog = np.log(float(ApplicantIncome) + float(CoapplicantIncome) if (float(ApplicantIncome)+float(CoapplicantIncome))>0 else 1.0)
    except:
        totalincomelog = 0.0
    try:
        LoanAmountlog = np.log(float(LoanAmount) if float(LoanAmount) > 0 else 1.0)
    except:
        LoanAmountlog = 0.0
    try:
        Loan_Amount_Termlog = np.log(float(Loan_Amount_Term) if float(Loan_Amount_Term) > 0 else 1.0)
    except:
        Loan_Amount_Termlog = 0.0

    try:
        credit_num = float(credit)
        credit_flag = 1 if 800 <= credit_num <= 1000 else 0
    except:
        credit_flag = 0

    return [
        credit_flag, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
        male, married_yes, dependents_1, dependents_2, dependents_3, not_graduate, employed_yes, semiurban, urban
    ]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","").strip()
        if not email or not password:
            flash("Email and password are required.", "danger")
            return redirect(url_for("register"))
        if query_user_by_email(email):
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))
        db = get_db()
        pw_hash = generate_password_hash(password)
        db.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, pw_hash))
        db.commit()
        flash("Registered successfully. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","").strip()
        user = query_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            user_obj = User(user["id"], user["email"])
            login_user(user_obj)
            flash("Logged in successfully.", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("predict"))
        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        fields = {
            "gender": request.form.get("gender","Male"),
            "married": request.form.get("married","No"),
            "dependents": request.form.get("dependents","0"),
            "education": request.form.get("education","Graduate"),
            "employed": request.form.get("employed","No"),
            "credit": request.form.get("credit","700"),
            "area": request.form.get("area","Urban"),
            "ApplicantIncome": request.form.get("ApplicantIncome","5000"),
            "CoapplicantIncome": request.form.get("CoapplicantIncome","0"),
            "LoanAmount": request.form.get("LoanAmount","100"),
            "Loan_Amount_Term": request.form.get("Loan_Amount_Term","360"),
        }
        try:
            float(fields["ApplicantIncome"])
            float(fields["CoapplicantIncome"])
            float(fields["LoanAmount"])
            float(fields["Loan_Amount_Term"])
            float(fields["credit"])
        except ValueError:
            flash("Please enter valid numeric values.", "danger")
            return redirect(url_for("predict"))

        features = preprocess_data(
            fields["gender"], fields["married"], fields["dependents"], fields["education"],
            fields["employed"], fields["credit"], fields["area"],
            fields["ApplicantIncome"], fields["CoapplicantIncome"], fields["LoanAmount"], fields["Loan_Amount_Term"]
        )

        if model is None:
            flash("Prediction model not available.", "warning")
            return redirect(url_for("predict"))

        try:
            pred = model.predict([features])
            pred_str = str(pred[0]).strip()
            approved = (pred_str == "Y" or pred_str.lower().startswith("y") or pred_str == "Yes" or pred_str == "1")
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
            return redirect(url_for("predict"))

        return render_template("predict.html", result=True, approved=approved, inputs=fields)
    return render_template("predict.html", result=False)

CHAT_QUESTIONS = [
    "What is your gender? (Male/Female)",
    "Are you married? (Yes/No)",
    "How many dependents do you have? (0/1/2/3+)",
    "What is your education level? (Graduate/Not Graduate)",
    "Are you self-employed? (Yes/No)",
    "What is your monthly applicant income?",
    "What is your monthly co-applicant income?",
    "What is the loan amount you are requesting?",
    "What is the loan amount term (in days)?",
    "What is your credit history score? (300-850)",
    "What is your area? (Urban/Semiurban/Rural)"
]

@app.route("/chatbot", methods=["GET","POST"])
@login_required
def chatbot():
    if "chat_step" not in session:
        session["chat_step"] = -1
        session["responses"] = {}

    if request.method == "POST":
        user_input = request.form.get("user_input","").strip()
        if session["chat_step"] == -1:
            if user_input.lower().startswith("y"):
                session["chat_step"] = 0
                session.modified = True
                return jsonify({"reply": "Great! " + CHAT_QUESTIONS[0], "step": session["chat_step"]})
            else:
                return jsonify({"reply": "Type 'Yes' when you are ready to begin.", "step": -1})

        step = session["chat_step"]
        if 0 <= step < len(CHAT_QUESTIONS):
            session["responses"][str(step)] = user_input
            session["chat_step"] = step + 1
            session.modified = True
            if session["chat_step"] < len(CHAT_QUESTIONS):
                return jsonify({"reply": CHAT_QUESTIONS[session["chat_step"]], "step": session["chat_step"]})
            else:
                r = session["responses"]
                try:
                    gender = r.get("0","Male")
                    married = r.get("1","No")
                    dependents = r.get("2","0")
                    education = r.get("3","Graduate")
                    self_employed = r.get("4","No")
                    applicant_income = r.get("5","5000")
                    coapplicant_income = r.get("6","0")
                    loan_amount = r.get("7","100")
                    loan_term = r.get("8","360")
                    credit_history = r.get("9","700")
                    property_area = r.get("10","Urban")

                    features = preprocess_data(gender, married, dependents, education, self_employed,
                                               credit_history, property_area, applicant_income, coapplicant_income,
                                               loan_amount, loan_term)
                    pred = model.predict([features])
                    pred_str = str(pred[0]).strip()
                    approved = (pred_str == "Y" or pred_str.lower().startswith("y") or pred_str == "Yes" or pred_str == "1")
                    session.pop("chat_step", None)
                    session.pop("responses", None)
                    return jsonify({"reply": "Prediction: " + ("APPROVED 🎉" if approved else "REJECTED ⚠️"), "step": "done", "approved": approved})
                except Exception as e:
                    session.pop("chat_step", None)
                    session.pop("responses", None)
                    return jsonify({"reply": f"Error while predicting: {e}", "step": "done", "approved": False})
        else:
            session.pop("chat_step", None)
            session.pop("responses", None)
            return jsonify({"reply": "Session reset. Start again by typing 'Yes'.", "step": -1})

    session["chat_step"] = -1
    session["responses"] = {}
    return render_template("chatbot.html", question=CHAT_QUESTIONS[0], step=-1)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
