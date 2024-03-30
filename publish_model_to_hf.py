import torch
import mmengine
import argparse

from huggingface_hub import HfApi

api = HfApi()

parser = argparse.ArgumentParser(description='publish models', add_help=False)
parser.add_argument('--upload_folder', action='store_true', help='upload folder')
parser.add_argument('--folder_path', type=str, required=True, help='local folder path')
parser.add_argument('--repo_id', type=str, required=True, help='repo id')
parser.add_argument('--repo_type', type=str, required=True, help='repo type')

args = parser.parse_args()


if args.upload_folder:
    api.upload_folder(folder_path=args.folder_path, repo_id=args.repo_id, repo_type=args.repo_type)
else:
    raise NotImplementedError("Only folder upload is supported for now")

