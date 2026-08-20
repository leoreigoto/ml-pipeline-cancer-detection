from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os

# Get host path defined in docker-compose
HOST_PROJECT_PATH = os.environ.get('HOST_PROJECT_PATH', os.getcwd())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 7),
    'retries': 0,
}

with DAG(
    'export_model_to_huggingface',
    default_args=default_args,
    schedule_interval=None, # Only activate manually (Trigger)
    catchup=False,
    tags=['mlops', 'export']
) as dag:

    export_task = DockerOperator(
        task_id='export_to_pth',
        image='uff_breast_cancer_pytorch:latest',
        api_version='auto',
        auto_remove=True,
        command="python MODEL_export_to_hf.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-breast-cancer-network",
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
            'HF_TOKEN': Variable.get("hf_token_secret", default_var="NOT_SET"),  # Inset value in Airflow UI (Admin -> Variables)
            'GIT_PYTHON_REFRESH': 'quiet'
        },
        mounts=[
            # Save .pth in local folder 'mlflow_data'
            {
                'Source': os.path.join(HOST_PROJECT_PATH, 'mlflow_data'),
                'Target': '/mlflow_data',
                'Type': 'bind'
            }
        ],
        working_dir='/app'
    )