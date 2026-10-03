import joblib

model = joblib.load("model/spam_model.pkl")

print("Spam Email Classifier")
print("---------------------")

message = input("Enter an email or message: ").strip()

if not message:
    print("Please enter a message.")
else:
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0]

    if prediction == 1:
        print("Prediction: SPAM")
        print("Confidence:", round(probability[1] * 100, 2), "%")
    else:
        print("Prediction: NOT SPAM")
        print("Confidence:", round(probability[0] * 100, 2), "%")