from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import shap
import numpy as np
import joblib
import os

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="ML based Loan Default Risk Prediction and Loan Analysis System",
    version="2.0.0"
)


# =========================================================
# FRONTEND
# =========================================================

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "models/loan_default_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("XGBoost model loaded successfully!")

except Exception as e:
    print("ERROR loading model:", e)
    model = None


# =========================================================
# LOAN APPLICATION INPUT
# =========================================================

class LoanApplication(BaseModel):

    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float

    term_months: float

    int_rate: float
    installment: float

    annual_inc: float
    dti: float

    fico_score: float

    delinq_2yrs: float
    inq_last_6mths: float

    open_acc: float
    pub_rec: float

    revol_bal: float
    revol_util: float

    total_acc: float

    grade: str
    sub_grade: str

    home_ownership: str
    verification_status: str
    purpose: str

    emp_length_years: float

    application_type: str


# =========================================================
# EMI REQUEST
# =========================================================

class EMIRequest(BaseModel):

    loan_amount: float
    annual_interest_rate: float
    tenure_years: int
    monthly_income: float


# =========================================================
# WHAT-IF REQUEST
# =========================================================

class WhatIfRequest(BaseModel):

    loan_amount: float
    annual_interest_rate: float
    tenure_months: int
    monthly_income: float

    fico_score: float
    dti: float


# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return FileResponse(
        "frontend/index.html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "error",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "XGBoost"
    }


# =========================================================
# LOAN PREDICTION
# =========================================================

@app.post("/predict")
def predict_loan(application: LoanApplication):

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="ML model could not be loaded."
        )

    try:

        # -------------------------------------------------
        # Convert input to dictionary
        # -------------------------------------------------

        data = application.model_dump()

        # -------------------------------------------------
        # Convert to DataFrame
        # -------------------------------------------------

        input_df = pd.DataFrame([data])

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        # -------------------------------------------------
        # DEFAULT PROBABILITY
        # -------------------------------------------------

        default_probability = probability * 100

        # -------------------------------------------------
        # RISK CATEGORY
        # -------------------------------------------------

        if default_probability < 30:

            risk_category = "Low"

        elif default_probability < 60:

            risk_category = "Medium"

        else:

            risk_category = "High"

        # -------------------------------------------------
        # CREDIT RISK SCORE
        # 300 - 900
        # -------------------------------------------------

        risk_score = int(
            round(
                900 - (probability * 600)
            )
        )

        risk_score = max(
            300,
            min(900, risk_score)
        )

        # -------------------------------------------------
        # LOAN DECISION
        # -------------------------------------------------

        if risk_category == "Low":

            loan_decision = "Likely Approved"

        elif risk_category == "Medium":

            loan_decision = "Review Required"

        else:

            loan_decision = "High Risk"

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "prediction": prediction,

            "default_probability": round(
                default_probability,
                2
            ),

            "risk_category": risk_category,

            "credit_risk_score": risk_score,

            "loan_decision": loan_decision,

            "model": "XGBoost"

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# SHAP EXPLAINABLE AI
# =========================================================

@app.post("/explain")
def explain_prediction(application: LoanApplication):

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="ML model could not be loaded."
        )

    try:

        # -------------------------------------------------
        # Convert input to DataFrame
        # -------------------------------------------------

        input_df = pd.DataFrame(
            [application.model_dump()]
        )

        # -------------------------------------------------
        # CHECK PIPELINE
        # -------------------------------------------------

        if not hasattr(model, "named_steps"):

            raise Exception(
                "Saved model is not a Pipeline. "
                "SHAP explanation requires the trained "
                "Pipeline containing preprocessing and XGBoost."
            )

        # -------------------------------------------------
        # GET PREPROCESSOR AND CLASSIFIER
        # -------------------------------------------------

        classifier = model[-1]
        preprocessor = model[:-1]

        # -------------------------------------------------
        # TRANSFORM INPUT
        # -------------------------------------------------

        transformed_input = preprocessor.transform(
            input_df
        )

        # -------------------------------------------------
        # FEATURE NAMES
        # -------------------------------------------------

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
                .tolist()
            )

        except Exception:

            feature_names = [
                f"Feature {i}"
                for i in range(
                    transformed_input.shape[1]
                )
            ]

        # -------------------------------------------------
        # CONVERT SPARSE MATRIX TO NUMPY
        # -------------------------------------------------

        if hasattr(
            transformed_input,
            "toarray"
        ):

            transformed_input = (
                transformed_input.toarray()
            )

        # -------------------------------------------------
        # CREATE SHAP EXPLAINER
        # -------------------------------------------------

        explainer = shap.TreeExplainer(
            classifier
        )

        # -------------------------------------------------
        # CALCULATE SHAP VALUES
        # -------------------------------------------------

        shap_values = explainer.shap_values(
            transformed_input,
            check_additivity=False
        )

        # -------------------------------------------------
        # HANDLE DIFFERENT SHAP OUTPUT FORMATS
        # -------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            shap_array = shap_values[1][0]

        else:

            shap_array = np.asarray(
                shap_values
            )

            if shap_array.ndim == 1:

                shap_array = shap_array

            elif shap_array.ndim == 2:

                shap_array = shap_array[0]

            elif shap_array.ndim == 3:

                shap_array = shap_array[
                    0,
                    :,
                    1
                ]

            else:

                raise Exception(
                    "Unexpected SHAP output shape: "
                    + str(shap_array.shape)
                )

        # -------------------------------------------------
        # CREATE CONTRIBUTION LIST
        # -------------------------------------------------

        contributions = []

        for name, value in zip(
            feature_names,
            shap_array
        ):

            value = float(value)

            clean_name = (
                name
                .replace("num__", "")
                .replace("cat__", "")
            )

            # Make feature names easier to read
            clean_name = (
                clean_name
                .replace("_", " ")
                .title()
            )

            if value > 0:

                impact = "Increases Default Risk"

            else:

                impact = "Decreases Default Risk"

            contributions.append({

                "feature": clean_name,

                "shap_value": round(
                    value,
                    6
                ),

                "impact": impact

            })

        # -------------------------------------------------
        # SORT BY ABSOLUTE IMPACT
        # -------------------------------------------------

        contributions.sort(
            key=lambda x: abs(
                x["shap_value"]
            ),
            reverse=True
        )

        # -------------------------------------------------
        # TOP 10 FEATURES
        # -------------------------------------------------

        top_features = (
            contributions[:10]
        )

        # -------------------------------------------------
        # RISK FACTORS
        # Positive SHAP = increases risk
        # -------------------------------------------------

        risk_factors = [

            x
            for x in top_features
            if x["shap_value"] > 0

        ]

        # -------------------------------------------------
        # POSITIVE FACTORS
        # Negative SHAP = decreases risk
        # -------------------------------------------------

        positive_factors = [

            x
            for x in top_features
            if x["shap_value"] < 0

        ]

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "explanation": top_features,

            "risk_factors": risk_factors,

            "positive_factors": positive_factors

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# EMI CALCULATOR
# =========================================================

@app.post("/calculate-emi")
def calculate_emi(request: EMIRequest):

    # -----------------------------------------------------
    # Input values
    # -----------------------------------------------------

    principal = float(
        request.loan_amount
    )

    annual_rate = float(
        request.annual_interest_rate
    )

    tenure_years = int(
        request.tenure_years
    )

    monthly_income = float(
        request.monthly_income
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if principal <= 0:

        raise HTTPException(
            status_code=400,
            detail="Loan amount must be greater than 0"
        )

    if annual_rate < 0:

        raise HTTPException(
            status_code=400,
            detail="Interest rate cannot be negative"
        )

    if tenure_years <= 0:

        raise HTTPException(
            status_code=400,
            detail="Tenure must be greater than 0"
        )

    if monthly_income <= 0:

        raise HTTPException(
            status_code=400,
            detail="Monthly income must be greater than 0"
        )

    # -----------------------------------------------------
    # EMI Calculation
    # -----------------------------------------------------

    monthly_rate = (
        annual_rate / 12 / 100
    )

    total_months = (
        tenure_years * 12
    )

    # -----------------------------------------------------
    # Zero Interest
    # -----------------------------------------------------

    if monthly_rate == 0:

        emi = (
            principal / total_months
        )

    else:

        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** total_months
        ) / (
            (1 + monthly_rate) ** total_months - 1
        )

    emi = float(emi)

    # -----------------------------------------------------
    # EMI TO INCOME RATIO
    # -----------------------------------------------------

    emi_ratio = (
        emi / monthly_income
    ) * 100

    # -----------------------------------------------------
    # AFFORDABILITY
    # -----------------------------------------------------

    if emi_ratio <= 30:

        affordability = "Comfortable"

        affordability_message = (
            "EMI is comfortably affordable "
            "relative to monthly income."
        )

    elif emi_ratio <= 40:

        affordability = "Moderate"

        affordability_message = (
            "EMI is manageable but may put "
            "some pressure on monthly finances."
        )

    else:

        affordability = "Risky"

        affordability_message = (
            "EMI is high relative to monthly income "
            "and may be difficult to repay."
        )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "loan_amount": round(
            principal,
            2
        ),

        "annual_interest_rate": round(
            annual_rate,
            2
        ),

        "tenure_years": tenure_years,

        "monthly_emi": round(
            emi,
            2
        ),

        "monthly_income": round(
            monthly_income,
            2
        ),

        "emi_to_income_ratio": round(
            emi_ratio,
            2
        ),

        "affordability": affordability,

        "message": affordability_message

    }


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

@app.post("/what-if")
def what_if(request: WhatIfRequest):

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="ML model could not be loaded."
        )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if request.loan_amount <= 0:

        raise HTTPException(
            status_code=400,
            detail="Loan amount must be greater than 0"
        )

    if request.annual_interest_rate < 0:

        raise HTTPException(
            status_code=400,
            detail="Interest rate cannot be negative"
        )

    if request.tenure_months <= 0:

        raise HTTPException(
            status_code=400,
            detail="Tenure must be greater than 0"
        )

    if request.monthly_income <= 0:

        raise HTTPException(
            status_code=400,
            detail="Monthly income must be greater than 0"
        )

    # -----------------------------------------------------
    # EMI CALCULATION
    # -----------------------------------------------------

    principal = float(
        request.loan_amount
    )

    monthly_rate = (
        request.annual_interest_rate
        / 12
        / 100
    )

    months = int(
        request.tenure_months
    )

    if monthly_rate == 0:

        emi = principal / months

    else:

        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** months
        ) / (
            (1 + monthly_rate) ** months - 1
        )

    emi = float(emi)

    # -----------------------------------------------------
    # EMI AFFORDABILITY
    # -----------------------------------------------------

    emi_ratio = (
        emi / request.monthly_income
    ) * 100

    if emi_ratio <= 30:

        affordability = "Comfortable"

    elif emi_ratio <= 40:

        affordability = "Moderate"

    else:

        affordability = "Risky"

    # -----------------------------------------------------
    # ANNUAL INCOME
    # -----------------------------------------------------

    annual_income = (
        request.monthly_income * 12
    )

    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    input_data = {

        "loan_amnt":
            request.loan_amount,

        "funded_amnt":
            request.loan_amount,

        "funded_amnt_inv":
            request.loan_amount,

        "term_months":
            request.tenure_months,

        "int_rate":
            request.annual_interest_rate,

        "installment":
            emi,

        "annual_inc":
            annual_income,

        "dti":
            request.dti,

        "fico_score":
            request.fico_score,

        "delinq_2yrs":
            0,

        "inq_last_6mths":
            0,

        "open_acc":
            8,

        "pub_rec":
            0,

        "revol_bal":
            0,

        "revol_util":
            0,

        "total_acc":
            15,

        "grade":
            "B",

        "sub_grade":
            "B2",

        "home_ownership":
            "RENT",

        "verification_status":
            "Verified",

        "purpose":
            "debt_consolidation",

        "emp_length_years":
            5,

        "application_type":
            "Individual"
    }

    input_df = pd.DataFrame(
        [input_data]
    )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    try:

        probability = float(
            model.predict_proba(
                input_df
            )[0][1]
        )

        prediction = int(
            model.predict(
                input_df
            )[0]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    default_probability = (
        probability * 100
    )

    # -----------------------------------------------------
    # RISK CATEGORY
    # -----------------------------------------------------

    if default_probability < 30:

        risk_category = "Low"

    elif default_probability < 60:

        risk_category = "Medium"

    else:

        risk_category = "High"

    # -----------------------------------------------------
    # CREDIT RISK SCORE
    # -----------------------------------------------------

    credit_risk_score = int(
        round(
            900 - probability * 600
        )
    )

    credit_risk_score = max(
        300,
        min(900, credit_risk_score)
    )

    # -----------------------------------------------------
    # LOAN DECISION
    # -----------------------------------------------------

    if risk_category == "Low":

        loan_decision = "Likely Approved"

    elif risk_category == "Medium":

        loan_decision = "Review Required"

    else:

        loan_decision = "High Risk"

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "loan_amount": round(
            principal,
            2
        ),

        "interest_rate": round(
            request.annual_interest_rate,
            2
        ),

        "tenure_months": months,

        "monthly_emi": round(
            emi,
            2
        ),

        "emi_to_income_ratio": round(
            emi_ratio,
            2
        ),

        "affordability": affordability,

        "default_probability": round(
            default_probability,
            2
        ),

        "risk_category":
            risk_category,

        "credit_risk_score":
            credit_risk_score,

        "loan_decision":
            loan_decision,

        "prediction":
            prediction,

        "model":
            "XGBoost"

    }


# =========================================================
# LOAN RECOMMENDATION
# =========================================================

@app.post("/loan-recommendation")
def loan_recommendation(request: WhatIfRequest):

    # -----------------------------------------------------
    # Calculate current EMI
    # -----------------------------------------------------

    principal = float(
        request.loan_amount
    )

    monthly_rate = (
        request.annual_interest_rate
        / 12
        / 100
    )

    months = int(
        request.tenure_months
    )

    if monthly_rate == 0:

        emi = principal / months

    else:

        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** months
        ) / (
            (1 + monthly_rate) ** months - 1
        )

    # -----------------------------------------------------
    # EMI RATIO
    # -----------------------------------------------------

    emi_ratio = (
        emi / request.monthly_income
    ) * 100

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    if emi_ratio <= 30:

        recommendation = (
            "Current loan amount is affordable."
        )

        recommended_loan_amount = principal

        recommended_tenure = months

    elif emi_ratio <= 40:

        recommendation = (
            "Consider a slightly longer tenure "
            "to reduce monthly EMI."
        )

        recommended_loan_amount = principal

        recommended_tenure = min(
            months + 12,
            84
        )

    else:

        recommendation = (
            "Loan EMI is high relative to income. "
            "Consider reducing the loan amount or "
            "increasing the tenure."
        )

        recommended_loan_amount = (
            principal * 0.80
        )

        recommended_tenure = min(
            months + 24,
            84
        )

    # -----------------------------------------------------
    # Recommended EMI
    # -----------------------------------------------------

    recommended_rate = (
        request.annual_interest_rate
        / 12
        / 100
    )

    recommended_months = int(
        recommended_tenure
    )

    if recommended_rate == 0:

        recommended_emi = (
            recommended_loan_amount
            / recommended_months
        )

    else:

        recommended_emi = (
            recommended_loan_amount
            * recommended_rate
            * (1 + recommended_rate)
            ** recommended_months
        ) / (
            (1 + recommended_rate)
            ** recommended_months - 1
        )

    # -----------------------------------------------------
    # Recommended EMI Ratio
    # -----------------------------------------------------

    recommended_emi_ratio = (
        recommended_emi
        / request.monthly_income
    ) * 100

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "current_loan_amount":
            round(
                principal,
                2
            ),

        "current_emi":
            round(
                emi,
                2
            ),

        "current_emi_ratio":
            round(
                emi_ratio,
                2
            ),

        "recommended_loan_amount":
            round(
                recommended_loan_amount,
                2
            ),

        "recommended_tenure_months":
            recommended_months,

        "recommended_emi":
            round(
                recommended_emi,
                2
            ),

        "recommended_emi_ratio":
            round(
                recommended_emi_ratio,
                2
            ),

        "recommendation":
            recommendation

    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():

    comparison_path = (
        "models/model_comparison.csv"
    )

    comparison = []

    if os.path.exists(
        comparison_path
    ):

        try:

            df = pd.read_csv(
                comparison_path
            )

            comparison = df.to_dict(
                orient="records"
            )

        except Exception:

            comparison = []

    return {

        "best_model":
            "XGBoost",

        "model_file":
            MODEL_PATH,

        "model_loaded":
            model is not None,

        "comparison":
            comparison

    }


# =========================================================
# END
# =========================================================