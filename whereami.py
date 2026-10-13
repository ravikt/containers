import getpass
import os
import platform
import socket
import sys


def os_name() -> str:
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.system()


def gpu() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__}, no GPU visible"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


rows = {
    "container": os.environ.get("APPTAINER_NAME", "none (host)"),
    "os": os_name(),
    "kernel": platform.release(),
    "python": f"{platform.python_version()} ({sys.executable})",
    "user": getpass.getuser(),
    "host": socket.gethostname(),
    "cwd": os.getcwd(),
    "gpu": gpu(),
}
for key, value in rows.items():
    print(f"{key:<10} {value}")
