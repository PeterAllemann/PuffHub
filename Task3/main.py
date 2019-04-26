from Task3.keyword_spotter import keyword_spotter
from Task3.evaluation import print_final_statistics

# define number of samples to compare
top_n = 50


keyword_spotter(top_n)
print_final_statistics(top_n)
