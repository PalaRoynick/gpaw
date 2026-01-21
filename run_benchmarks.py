from gpaw.benchmark import benchmark_main, get_benchmarks

benchmarks = get_benchmarks()
print(benchmarks)

benchmark_main("C60_pw")