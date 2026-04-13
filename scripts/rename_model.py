import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI  = "http://127.0.0.1:5000"
OLD_NAME    = "churn_xgboost"
NEW_NAME    = "churn_predictor"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# Step 1 — copy all versions to the new name
versions = client.search_model_versions(f"name='{OLD_NAME}'")

for v in versions:
    client.copy_model_version(
        src_model_uri = f"models:/{OLD_NAME}/{v.version}",
        dst_name      = NEW_NAME,
    )
    print(f"Copied version {v.version} → {NEW_NAME}")

# Step 2 — archive then delete the old registered model
for v in versions:
    client.transition_model_version_stage(
        name    = OLD_NAME,
        version = v.version,
        stage   = "Archived",
    )

client.delete_registered_model(OLD_NAME)
print(f"Deleted old model '{OLD_NAME}'. Done.")