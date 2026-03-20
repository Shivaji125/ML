from locust import HttpUser, between, task

valid_payload = {
    "CreditScore": 650,
    "Age": 40,
    "Balance": 75000.0,
    "EstimatedSalary": 60000.0,
    "Geography": "France",
    "Gender": "Male",
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "Tenure": 5,
}


class ChurnUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self):
        self.client.post("/predict", json=valid_payload)

    @task(1)
    def health_check(self):
        self.client.get("/health")
