"""Airflow DAG: download -> preprocess -> train -> evaluate -> build_index.

Each task is a BashOperator calling the matching `python -m src.*` module so the
pipeline is identical locally, in Docker, and in CI. Trigger manually or on schedule.
Set REPO env var / volume to the repo checkout (see ../docker-compose.yml).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

REPO = "{{ dag_run.conf.get('repo', '/opt/repo') if dag_run and dag_run.conf else '/opt/repo' }}"
SUBSET = "{{ dag_run.conf.get('max_samples', 200) if dag_run and dag_run.conf else 200 }}"

default_args = {"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="shape_retrieval",
    description="ModelNet10 3D shape retrieval pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["mlops", "3d-retrieval"],
) as dag:
    download = BashOperator(
        task_id="download_data",
        bash_command=f"cd /opt/repo && python -m src.data_loader --max-samples-per-class {SUBSET}",
    )
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /opt/repo && python -m src.preprocess",
    )
    train_baseline = BashOperator(
        task_id="train_baseline",
        bash_command="cd /opt/repo && python -m src.train --model baseline",
    )
    train_pointnet = BashOperator(
        task_id="train_pointnet",
        bash_command="cd /opt/repo && python -m src.train --model pointnet --epochs 10",
    )
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="cd /opt/repo && python -m src.evaluate",
    )
    build_index = BashOperator(
        task_id="build_index",
        bash_command="cd /opt/repo && python -m src.build_index --model pointnet",
    )

    download >> preprocess >> [train_baseline, train_pointnet] >> evaluate >> build_index
