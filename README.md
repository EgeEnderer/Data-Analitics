# Portfolio

## Revenue Analytics and Forecasting Project

### Overview

This project was conceived to furnish the senior leadership team with a transparent and data-driven understanding of historical revenue performance and to deliver robust forward projections. 
It encompasses comprehensive data preparation, exploratory analysis, statistical modelling and interactive reporting. 
The forecasting component leverages both ARIMA and Holt-Winters methods to produce reliable estimates for the next three and twelve months. 
Outputs are delivered via a Power BI dashboard and an executive presentation, ensuring that stakeholders can engage with key insights and monitor revenue trends in real time.

### Data Preparation and Interpolation

A year’s worth of daily records for average revenue per user (ARPU) across seven markets and total revenue in euros served as the primary dataset. 
Initial processing involved converting date strings into a proper datetime index and addressing missing entries. 
Spline interpolation of order five was applied to smooth gaps in the ARPU series for each market, preserving the integrity of the time series. 
The interpolated series were saved to file and visualised to confirm continuity and consistency before further analysis.

### Exploratory Time Series Analysis

Exploratory steps included plotting the raw and interpolated series, decomposing total revenue into trend, seasonal and residual components, and assessing stationarity. 
Augmented Dickey-Fuller tests were applied to both the original and differenced series to determine the need for transformations. 
Autocorrelation and partial autocorrelation functions provided guidance on appropriate model orders and seasonal lags. 
These exploratory analyses informed the parameterisation of both ARIMA and Holt-Winters models.

### ARIMA Forecasting

An automated ARIMA routine selected optimal orders for the non-seasonal and seasonal components by iteratively testing combinations against stationarity criteria and information-criterion metrics. 
The final model was fitted to the daily total-revenue series and generated point forecasts for a 90-day horizon. 
Confidence intervals at the 70, 80 and 95 per cent levels were computed, allowing decision-makers to appreciate the range of potential outcomes. 
Model performance was evaluated using mean squared error, root mean squared error and mean absolute error, and the fitted values were compared visually to the historical data to validate accuracy.

### Holt-Winters Exponential Smoothing

Multiple Holt-Winters specifications were assessed, including additive and multiplicative forms of trend and seasonality with damped-trend enabled. 
Each configuration was fitted to the total-revenue series, and performance metrics guided the selection of the best model. 
The chosen specification produced a forecast that complemented the ARIMA output, providing an alternate perspective on future revenue dynamics. 
Residuals of the best Holt-Winters model were analysed to confirm the absence of autocorrelation and to ensure that model assumptions held.

### Market Contribution Analysis

In addition to aggregate revenue forecasts, the project examined market-level ARPU trends to determine which regions contribute most significantly to total revenue. 
This analysis highlighted high-value markets and informed recommendations for resource allocation and strategic focus. 
Findings were incorporated into both the dashboard and the executive presentation to guide leadership in identifying priority markets.

### Reporting and Executive Presentation

A Power BI dashboard was developed to deliver an interactive view of daily and monthly revenue evolution, ARPU variations by market and forecast projections. 
The interface allows users to filter by market, switch between short-term and long-term outlooks and visualise confidence intervals. 
An accompanying executive presentation distils the analysis into concise slides, outlining key observations, forecast narratives and strategic considerations such as pricing pressures, customer churn,
market campaigns and macroeconomic influences.

### Technical Environment

All data processing and modelling were performed in Python 3. 
Core libraries included pandas and numpy for data manipulation, matplotlib for visualisation, statsmodels and pmdarima for time series modelling, and scikit-learn for performance evaluation. 
The Power BI desktop application served as the platform for business intelligence reporting and dashboard authoring.

### Deliverables

The project deliverables comprise a cleaned and validated dataset, interpolation outputs, exploratory diagnostic plots, ARIMA and Holt-Winters forecasts with confidence intervals, market contribution insights,
an interactive Power BI dashboard and an executive slide deck. Together, these assets equip the leadership team with actionable intelligence to drive data-informed decisions and sustain revenue growth.
