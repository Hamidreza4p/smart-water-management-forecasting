# Data

This project uses the Bilancino Lake data from the [Acea Smart Water Analytics competition](https://www.kaggle.com/competitions/acea-water-prediction/data).

Download the competition data according to Kaggle's access terms. The evaluation script expects a prepared file at:

```text
data/lakes_fe.csv
```

Expected columns:

| Column | Description |
|---|---|
| `Date` | Consecutive integer day offset beginning at 0 |
| `Mean_Rainfall` | Prepared daily mean rainfall |
| `Mean_Temp` | Prepared daily mean temperature |
| `Actual_Flow_Rate` | Prepared daily flow-rate target |
| `Actual_Lake_Level` | Prepared daily lake-level target |

The supplied prepared dataset contains 6,603 observations. Offset 0 is interpreted as June 3, 2002, producing a final date of June 30, 2020.

The CSV is intentionally excluded from Git because redistribution rights are governed by the source dataset's terms.

