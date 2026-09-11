import torch


def configure_cpu():
    """
    为当前项目配置 PyTorch CPU 线程。

    当前机器：
        Intel i5-1035G1
        4 physical cores
        8 logical processors

    Benchmark 最优：
        intra-op = 8
        inter-op = 1
    """

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    print(
        f"PyTorch CPU threads: "
        f"{torch.get_num_threads()}"
    )

    print(
        f"PyTorch interop threads: "
        f"{torch.get_num_interop_threads()}"
    )