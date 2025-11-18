# utils/chatbot_engine.py
import random

def get_chatbot_reply(message):
    msg = (message or "").lower().strip()
    if not msg:
        return "Hi — how can I help with your loan question? (e.g., 'What affects my credit score?')"

    # Simple pattern matching / rules
    if "credit score" in msg:
        return ("A credit score reflects your creditworthiness. Pay bills on time, reduce credit utilization, "
                "and check your credit report for errors. Improving these factors typically raises your score.")
    if "emi" in msg or "installment" in msg:
        return ("EMI = (P * r * (1+r)^n) / ((1+r)^n - 1) where P=principal, r=monthly interest rate, n=no. of months."
                " You can provide numbers and I can compute an EMI for you.")
    if "why rejected" in msg or "rejected" in msg:
        return ("Common reasons: low credit score, high debt-to-income, insufficient income, or incomplete documentation.")
    if "what is dti" in msg or "debt to income" in msg:
        return ("Debt-to-income ratio (DTI) = (monthly debt payments / monthly gross income). Lenders prefer a lower DTI.")
    # fallback small talk
    fallback = [
        "Can you provide more details (credit score, income, loan amount)?",
        "I can help explain the decision or compute EMI — what would you like?",
        "If you want, type 'EMI 500000 10 7' (principal, years, annual_rate) to compute EMI."
    ]
    return random.choice(fallback)
