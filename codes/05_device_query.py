import torch


def byte_to_gb(x: int) -> float:
    return x / (1024 ** 3)

def print_device_summary(device_id: int):
    p = torch.cuda.get_device_properties(device_id)

    print(f"\n[GPU {device_id}] {p.name}")
    print(f"  compute capability : {p.major}.{p.minor}")
    print(f"  total memory       : {byte_to_gb(p.total_memory):.2f} GB")
    print(f"  SM count           : {p.multi_processor_count}")
    print(f"  warp size          : {p.warp_size}")

    alloc = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    print(f"  mem allocated      : {byte_to_gb(alloc):.4f} GB")
    print(f"  mem reserved       : {byte_to_gb(reserved):.4f} GB")
    print(f"  mem free(approx)   : {byte_to_gb(p.total_memory - reserved):.2f} GB")


def smoke_test(): # 최소 동작
    # CUDA 디바이스 접근 가능
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = a + b
    # 커널/연산이 GPU에서 실제 실행됨
    torch.cuda.synchronize()
    # 동기화까지 에러 없이 완료됨
    print("\nCUDA smoke test: OK")
    print(f"  sum: {c.sum().item():.4f}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    n = torch.cuda.device_count()
    print(f"CUDA devices: {n}")

    for i in range(n):
        print_device_summary(i)

    smoke_test()


if __name__ == "__main__":
    main()
