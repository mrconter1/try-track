import pstats
from pstats import SortKey

try:
    p = pstats.Stats('profile.pstats')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
except FileNotFoundError:
    print("Error: 'profile.pstats' not found.")
    print("Please run the profiler command first: python -m cProfile -o profile.pstats incremental_map_builder.py")
