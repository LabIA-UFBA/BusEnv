# Rates experiment — mean onboard load arriving at the stop — method comparison

Instances: **334 stops** | methods: timesfm. Metrics computed only where y_true is observed; MAPE excludes y_true = 0 points.

## Overall metrics (all instances pooled, weighted by points)

| method   |   n_points |     mse |    mae |   mape_pct |
|:---------|-----------:|--------:|-------:|-----------:|
| timesfm  |     187064 | 84.3122 | 5.9761 |    41.7342 |

## Win rates (per stops: lowest error wins; ties split equally)

### MSE

- **timesfm**: won 100.0% of stops

![pie](pie_mse.png)

### MAE

- **timesfm**: won 100.0% of stops

![pie](pie_mae.png)

### MAPE (%)

- **timesfm**: won 100.0% of stops

![pie](pie_mape_pct.png)
