import os


def set_gpu_power_limit_if_needed(pw: int = 250) -> None:
    """
    Helper function, that sets GPU power limit if RTX 3090 is used

    Args:
        pw: The new power limit to set. Defaults to 250W.

    """

    stream = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv")
    gpu_list = stream.read()
    if "NVIDIA GeForce RTX 3090" in gpu_list:
        os.system("sudo nvidia-smi -pm 1")
        os.system(f"sudo nvidia-smi -pl {pw}")
