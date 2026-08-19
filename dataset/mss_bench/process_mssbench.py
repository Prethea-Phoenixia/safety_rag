import os.path
from huggingface_hub import snapshot_download

snapshot_download(repo_id="kzhou35/mssbench", repo_type="dataset", local_dir=os.path.dirname(__file__))
