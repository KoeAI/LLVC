# download models from KoeAI/llvc on Huggingface Hub
from huggingface_hub import snapshot_download

# download models from KoeAI/llvc on Huggingface Hub
snapshot_download(repo_id="KoeAI/llvc",
                  local_dir='llvc_models')
