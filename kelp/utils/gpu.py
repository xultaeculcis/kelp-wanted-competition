import os


def set_gpu_power_limit_if_needed() -> None:
    """Helper function, that sets GPU power limit if RTX 3090 is used"""

    stream = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv")
    gpu_list = stream.read()
    if "NVIDIA GeForce RTX 3090" in gpu_list:
        os.system("sudo nvidia-smi -pm 1")
        os.system("sudo nvidia-smi -pl 250")
