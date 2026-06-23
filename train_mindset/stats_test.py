from scipy.stats import wilcoxon

# # The exact fold-by-fold accuracies we generated earlier
# text_only = [0.59, 0.58, 0.88, 0.86, 0.90]
# trimodal_peak = [0.81, 0.58, 0.88, 0.85, 0.90]  # Text + Face + HRV
# full_baseline = [0.71, 0.58, 0.88, 0.89, 0.90]  # Text + Audio + Face + HRV

# print("--- Wilcoxon Signed-Rank Test Results ---")

# # 1. Does adding Face and HRV significantly improve Text alone?
# stat1, p1 = wilcoxon(text_only, trimodal_peak)
# print(f"Text Only vs. Trimodal Peak (Face+HRV) -> p-value: {p1:.4f}")

# # 2. Is the Full Architecture significantly different from Text Only?
# stat2, p2 = wilcoxon(text_only, full_baseline)
# print(f"Text Only vs. Full 4-Modality -> p-value: {p2:.4f}")


# from scipy.stats import ttest_rel

# full = [0.78, 0.58, 0.87, 0.86, 0.90]

# ablation_1 = [0.81, 0.58, 0.88, 0.85, 0.90]
# ablation_2 = [0.57, 0.58, 0.88, 0.83, 0.90]
# ablation_3 = [0.57, 0.57, 0.88, 0.88, 0.90]
# ablation_4 = [0.59, 0.58, 0.88, 0.86, 0.90]
# ablation_5 = [0.71, 0.58, 0.88, 0.89, 0.90]

# for name, model in {
#     "No Audio + No Eye": ablation_1,
#     "Text+Face": ablation_2,
#     "Text+Audio": ablation_3,
#     "Text Only": ablation_4,
#     "No Eye": ablation_5
# }.items():

#     t_stat, p_value = wilcoxon(full, model)
#     print(f"{name}: p={p_value:.4f}")


import scipy.stats as stats

trimodal_acc = [0.760, 0.790, 0.830, 0.830, 0.810]
unimodal_acc = [0.690, 0.740, 0.780, 0.820, 0.780]

# Calculate differences
diffs = [t - u for t, u in zip(trimodal_acc, unimodal_acc)]
print("Differences:", diffs)

# Wilcoxon signed-rank test
stat, p_value = stats.wilcoxon(trimodal_acc, unimodal_acc)
print(f"Wilcoxon Statistic: {stat}")
print(f"p-value: {p_value}")



stat, p_value_one_sided = stats.wilcoxon(trimodal_acc, unimodal_acc, alternative='greater')
print(f"One-sided p-value: {p_value_one_sided}")