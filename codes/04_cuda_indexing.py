def cdiv(n: int, d: int) -> int:
    if d <= 0:
        raise ValueError("d must be positive")
    return (n + d - 1) // d


# 예시
N = 1000
block = 256
grid = cdiv(N, block)
print(grid)  # 4
