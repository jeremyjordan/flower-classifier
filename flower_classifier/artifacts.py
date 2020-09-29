import wandb


def list_runs(project: str = "flowers", entity: str = "jeremytjordan"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return [r.id for r in runs]


def list_run_files(run_id: str, project: str = "flowers", entity: str = "jeremytjordan"):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return [f.name for f in run.files()]


def download_model_checkpoint(
    checkpoint_name: str, run_id: str, download_dir: str = ".", project: str = "flowers", entity: str = "jeremytjordan"
):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    f = run.file(checkpoint_name)
    f.download(root=download_dir, replace=True)  # TODO find a better place to download to
    return f.name
