"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    data = pd.read_csv(data.csv_path)
    return TitanicSummary(
        row_count=data.shape[0],
        missing_by_column=data.isna().sum().to_dict(),
        adults_over_30_count=len(data[data["Age"] > 30]),
        mean_age_by_pclass=data.groupby('Pclass')['Age'].mean().to_dict(),
        survival_rate_by_pclass=data.groupby('Pclass')['Survived'].mean().to_dict(),
        highest_fares=data['Fare'].nlargest(5).to_list()
    )
# q = analyze_titanic(TitanicInput("titanic.csv"))
# print(q)