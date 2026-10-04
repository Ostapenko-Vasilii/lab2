"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделислите ките наблюдения по полу и для каждой группы вычорреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    na_values=["NA", "?"]
    """
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    target = 'MRI_Count'
    cols = features + [target]
    df = pd.read_csv(data.csv_path, sep='\t', na_values=["NA", "?"])
    # plt.figure(figsize=(14, 5))
    #
    # plt.subplot(1, 2, 1)

    fdf = df[df['Gender'] == 'Female']
    mdf = df[df['Gender'] == 'Male']
    # sns.heatmap(fdf[cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    # plt.subplot(1, 2, 2)
    # sns.heatmap(mdf[cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    # plt.tight_layout()
    # plt.show()
    #
    # sns.pairplot(
    #     df,
    #     x_vars=features,
    #     y_vars=[target],
    #     hue='Gender',
    #     kind='scatter',
    #     height=4,
    #     aspect=0.8
    # )
    # plt.show()

    men_mri_correlation = mdf[cols].corr(method='pearson')[target][features]
    women_mri_correlation = fdf[cols].corr(method='pearson')[target][features]
    all_correlations = pd.concat([women_mri_correlation, men_mri_correlation])
    return  BrainCorrelationSummary(
        men_count= mdf.shape[0],
        women_count= fdf.shape[0],
        women_mri_correlation=women_mri_correlation.to_dict(),
        men_mri_correlation=men_mri_correlation.to_dict(),
        strongest_mri_feature=str(all_correlations.abs().idxmax())
    )
# analyze_brain_correlations(BrainDataInput(csv_path='brainsize.txt'))