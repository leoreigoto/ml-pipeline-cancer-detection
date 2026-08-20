from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
import json
import os
import re


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'catchup': False,
}


# Name of the network created by Docker Compose.
# Default: foldername_default. But we are setting it to  mlops-breast-cancer-network in the yml.
# `docker network ls` in cmd -> returns the network name
DOCKER_NETWORK_ID = 'mlops-breast-cancer-network'


def parse_last_json(value):
    """
    Parses the last valid JSON object found in a string.
    This handles cases where the container prints logs before or after the JSON.
    """
    if not value:
        return {}

    # Try parsing the whole string directly
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # If that fails, regex search for the last JSON-like structure: { ... }
    # This regex looks for a bracket, anything inside, and a closing bracket
    try:
        match = re.search(r'(\{.*\})', value, re.DOTALL)
        if match:
            # If multiple JSONs exist, this usually grabs the outer one or the last one
            candidate = match.group(1)
            return json.loads(candidate)
    except Exception:
        pass

    # If all fails, raise error
    raise ValueError(f"Could not extract JSON from XCom value: {str(value)[:200]}...")

def get_host_path(relative_path):
    """
    Constructs the absolute path on the HOST machine.
    Relies on HOST_PROJECT_PATH env var set in docker-compose.
    """
    host_root = os.getenv("HOST_PROJECT_PATH")

    if not host_root:
        raise AirflowException(
            "Environment variable 'HOST_PROJECT_PATH' is missing. "
            "Please add 'HOST_PROJECT_PATH: ${PWD}' to your docker-compose.yml "
            "under x-airflow-common > environment."
        )

    # Clean up potential trailing slashes and join
    return os.path.join(host_root.strip(os.sep), relative_path.strip(os.sep))

with DAG(
    'containerized_ml_pipeline',
    default_args=default_args,
    schedule_interval=None,
    #render_template_as_native_obj=True,
    user_defined_filters={'from_json': parse_last_json},
) as dag:

    # 1. Train Model
    # Runs the training script inside a container
    train_model = DockerOperator(
        task_id='train_model',
        image='uff_breast_cancer_pytorch:latest',
        #container_name='task_train_model',
        api_version='auto',
        auto_remove=True,
        force_pull=False,
        command="python MODEL_main.py",
        do_xcom_push=True, # For Docker Operator we need to print the xcom args in the end of the file execution
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK_ID,
        # shm_size is the RAM size, might need adjustment for larger model / training set
        shm_size='3g',
        mount_tmp_dir=False,
        environment={
            # We use the container name 'mlflow-server' because we are on the same network
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
            'MLFLOW_EXPERIMENT_NAME': 'Default'
        },
        mounts=[
            Mount(source=get_host_path('mlflow_data'),target='/mlflow_data',type='bind')
        ]
    )

    # 2. Evaluate and Register
    evaluate_model = DockerOperator(
        task_id='evaluate_and_register_model',
        image='uff_breast_cancer_pytorch:latest',
        #container_name='task_evaluate_model',
        api_version='auto',
        auto_remove=True,
        # We use Jinja templates to parse the JSON XCom
        command="""
        python MODEL_evaluate_and_register.py \
        --model_uri {{ (task_instance.xcom_pull(task_ids='train_model') | from_json)['model_uri'] }} \
        --pipeline_uri {{ (task_instance.xcom_pull(task_ids='train_model') | from_json)['pipeline_uri'] }} \
        --model_name {{ (task_instance.xcom_pull(task_ids='train_model') | from_json)['model_name'] }}
        """,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK_ID,
        # shm_size is the RAM size, might need adjustment for larger model / eval set
        shm_size='3g',
        mount_tmp_dir=False,
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
        },
        mounts=[
            Mount(source=get_host_path('mlflow_data'), target='/mlflow_data', type='bind')
        ]
    )

    train_model >> evaluate_model