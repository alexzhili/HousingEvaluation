from mlflow.tracking import MlflowClient
client = MlflowClient()
display(client.list_experiments())

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)



model_name = f"OPS_mllib_lr"
model_uri = f"runs:/{run.info.run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)


from mlflow.tracking.client import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)