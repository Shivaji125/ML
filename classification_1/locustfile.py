from locust import HttpUser, task, between

valid_payload = {
    "CreditScore": 650,
    "Age": 40,
    "Balance": 75000,
    "EstimatedSalary": 60000,
    "Geography": "France",
    "Gender": "Male",
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "Tenure": 5
}

class ChurnUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json=valid_payload)