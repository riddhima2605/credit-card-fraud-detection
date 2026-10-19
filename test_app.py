"""Run:  python test_app.py   (from the fraud_app folder)"""
import io
import os
import pandas as pd
from app import app, FEATURES, ART_DIR

client = app.test_client()
SAMPLE = os.path.join(ART_DIR, "sample_transactions.csv")
passed = 0


def check(name, condition):
    global passed
    print(("PASS  " if condition else "FAIL  ") + name)
    if condition:
        passed += 1


def upload(data, filename="test.csv", url="/predict"):
    return client.post(url, data={"file": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


sample = pd.read_csv(SAMPLE)
good = sample.to_csv(index=False).encode()

# 1. pages load
check("home page loads", client.get("/").status_code == 200)
check("demo page loads and shows results", b"flagged as fraud" in client.get("/demo").data)

# 2. valid upload
r = upload(good)
check("valid CSV accepted", r.status_code == 200 and b"flagged as fraud" in r.data)

# 3. JSON API returns probability + prediction
r = upload(good, url="/api/predict")
j = r.get_json()
check("API returns probabilities", r.status_code == 200 and len(j["predictions"]) == len(sample))
check("probabilities are between 0 and 1", all(0 <= p["probability"] <= 1 for p in j["predictions"]))
check("predictions are Fraud/Genuine", {p["prediction"] for p in j["predictions"]} <= {"FRAUD", "Genuine"})

# 4. file without Class column still works
r = upload(sample.drop(columns=["Class"]).to_csv(index=False).encode())
check("CSV without Class column works", r.status_code == 200)

# 5. shuffled column order still gives the same answers (order is enforced)
shuffled = sample[FEATURES[::-1] + ["Class"]]
j2 = upload(shuffled.to_csv(index=False).encode(), url="/api/predict").get_json()
check("column order does not matter",
      [p["probability"] for p in j["predictions"]] == [p["probability"] for p in j2["predictions"]])

# 6. invalid inputs must not crash
check("empty file -> 400", upload(b"").status_code == 400)
check("no file -> 400", client.post("/predict", data={}).status_code == 400)
check("wrong extension -> 400", upload(good, "data.txt").status_code == 400)
check("garbage bytes -> 400", upload(b"\x00\x01\x02not,a,csv\xff").status_code == 400)
check("header only -> 400", upload((",".join(FEATURES) + "\n").encode()).status_code == 400)
check("missing columns -> 400", upload(sample.drop(columns=["V1", "V2"]).to_csv(index=False).encode()).status_code == 400)

bad = sample.copy().astype(object)
bad.loc[0, "Amount"] = "abc"
check("non-numeric value -> 400", upload(bad.to_csv(index=False).encode()).status_code == 400)

nan = sample.copy()
nan.loc[0, "V3"] = None
check("missing value (NaN) -> 400", upload(nan.to_csv(index=False).encode()).status_code == 400)

big = pd.concat([sample] * 130, ignore_index=True)  # 5200 rows > 5000 limit
check("too many rows -> 400", upload(big.to_csv(index=False).encode()).status_code == 400)

check("API with no file -> 400 JSON", client.post("/api/predict").get_json()["error"] != "")

print(f"\n{passed} checks passed")
