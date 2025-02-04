#!/usr/bin/env python
import os
import warnings
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

def push_model_to_hf_hub(
    local_model_path,
    repo_id,
    private=False,
    replace=False,
    revision="main",
    commit_message="Add model",
    hf_token=None
):
    """
    Loads a model and its tokenizer from a local directory and pushes both to the Hugging Face Hub.
    If the repository does not exist, it is created. If it exists and `replace` is False, an error is raised.
    If it exists and `replace` is True, a warning is issued and the content is replaced.
    """
    # Retrieve the Hugging Face token from parameter or environment
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("Please set the HF_TOKEN environment variable or provide the hf_token parameter.")

    # Initialize the HfApi
    api = HfApi()

    # Attempt to create the repository
    try:
        # exist_ok=False forces an error if the repo already exists.
        api.create_repo(repo_id, token=hf_token, private=private, exist_ok=False)
        print(f"Repository '{repo_id}' created successfully.")
    except Exception as exc:
        # Convert the exception to string to check for a 409 conflict
        # or messages like 'already exists'.
        error_str = str(exc)
        if "409" in error_str or "already exists" in error_str:
            if not replace:
                raise ValueError(
                    f"Repository '{repo_id}' already exists. "
                    "Set replace=True if you want to replace its content."
                )
            else:
                warnings.warn(
                    f"Repository '{repo_id}' already exists. Replacing its content as requested.",
                    UserWarning
                )
        else:
            # If it's a different error, re-raise it
            raise exc

    # Load the model and tokenizer from the local directory
    print("Loading model and tokenizer from local path ...")
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    
    # Push the model to the Hub
    print(f"Pushing model to '{repo_id}' ...")
    model.push_to_hub(repo_id, token=hf_token, private=private, revision=revision, commit_message=commit_message)
    
    # Push the tokenizer to the Hub
    print(f"Pushing tokenizer to '{repo_id}' ...")
    tokenizer.push_to_hub(repo_id, token=hf_token, private=private, revision=revision, commit_message=commit_message)
    
    print("Push complete.")

if __name__ == "__main__":
    fire.Fire(push_model_to_hf_hub)