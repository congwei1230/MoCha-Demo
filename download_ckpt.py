from huggingface_hub import hf_hub_download

local_dir = "YOUR LOCAL FOLDER"

ckpt_path = hf_hub_download(
    repo_id="CongWei1230/MoCha",
    filename="model.ckpt",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(ckpt_path)