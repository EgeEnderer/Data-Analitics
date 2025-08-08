# Portfolio

Welcome to my data analytics and project management portfolio. This repository showcases a collection of business-focused analytics projects that bring together data science, time series forecasting, reporting automation and strategic business analysis.

Each project is independent and designed to solve real-world business challenges using a blend of Python scripting, Excel modelling, Power BI dashboards and executive-level presentations. These solutions are tailored to improve operational efficiency, drive revenue insights and support data-driven decision-making across commercial and PMO contexts.

## Projects

### 1. Business Performance Analytics (2023–2027)

#### Overview

This project consolidates shipment and revenue records from 2023 through early 2027 to deliver a clear, data-driven view of business performance. The dataset includes 199 shipments across 18 customers in 17 countries and four product lines, generating approximately €25,077.71 in total revenue.

The analysis describes historical trends in shipment volume and revenue, identifies high-performing customers and product lines, uncovers seasonal patterns, and provides commercial recommendations to support future business planning.

#### Data Summary and Cleaning

The raw dataset, stored in Excel format, contained shipment-level details including product category, customer, shipment volume, origin country, revenue, and shipment date. Initial review revealed sparse data for 2023 and incomplete records beyond February 2027. These limitations were documented and excluded from forward-looking analysis.

Data cleaning steps included:
- Standardising date formats
- Validating and correcting revenue values
- Normalising country and product names
- Creating derived time fields for monthly and yearly aggregation

The final cleaned dataset is structured for time series, comparative, and seasonal analysis.

#### Analysis Highlights

Revenue and shipment counts rose steadily from 2024 to 2026, with approximately 10 per cent annual growth. Recurring monthly peaks were observed in May, September and October, while February remained consistently low. Economy Freight led revenue contribution, while Express Freight and Parcel services showed stable growth. A small number of customers generated most of the value, with select smaller accounts displaying high revenue per shipment. Germany, France and Hungary were top performers, while Japan and Poland showed exceptional profitability per unit shipped.

#### Strategic Recommendations

- Prioritise high-value markets: Japan, Poland, Germany, France and Spain  
- Reassess commercial strategies in lower-performing regions  
- Align operations with seasonal demand patterns  
- Balance profitability and cost across freight service tiers  
- Improve 2027 data completeness for stronger forecasting

#### Reporting and Communication

Delivered via:
- An interactive Power BI dashboard  
- An executive presentation highlighting key metrics, methods and decisions  
- Structured Excel workbooks with embedded documentation

#### Tools Used

Excel, Power BI, PowerPoint

---

### 2. Revenue Forecasting and Market Modelling (ARIMA & Holt-Winters)

- [Link to Overview of PowerBI Dasboard](https://github.com/EgeEnderer/Portfolio/blob/a257db9473af3619cfddd67ecc5e5cc3b489193a/Revenue%20Analytics/Revenue%20Analysis%20Dashboard.pdf)
- [Link to PowerBI Dasboard](https://github.com/EgeEnderer/Portfolio/blob/a257db9473af3619cfddd67ecc5e5cc3b489193a/Revenue%20Analytics/Revenue%20Analysis%20Dashboard.pbix)

#### Overview

This project supports leadership with reliable revenue forecasts across seven markets. It combines time series modelling, statistical analysis, and business reporting to deliver short-term (3 months) and long-term (12 months) outlooks using ARIMA and Holt-Winters models.

#### Data and Preparation

Daily revenue and ARPU data was cleaned and interpolated using spline functions. 

Data cleaning steps included:
- Standardising date formats
- Validating and correcting revenue values
- Creating derived time fields for monthly and yearly aggregation
- Interpolating missing values and imputing it with interpolated values
  
After preprocessing, data was analysed for trend and stationarity using decomposition and ADF tests.

#### Forecasting Approach

- **ARIMA**: Multiple configurations were tested using automated parameter tuning. Residuals were analysed to validate assumptions, and confidence intervals were computed. The best model was selected based on RMSE.
- **Holt-Winters**: Similarly, several configurations were explored, including different seasonal and trend components. The models underwent residual diagnostics and were tested against confidence intervals. Final model selection was based on lowest RMSE.

Forecasts were visualised with uncertainty bands and exported to CSV for integration into dashboards.

#### Tools Used

Excel, Python (pandas, NumPy, matplotlib, seaborn, statsmodels, pmdarima, scikit-learn ), Power BI, PowerPoint

#### Insights and Reporting

Revenue forecasts and market contribution trends were included in a Power BI dashboard and executive presentation. These deliverables support strategic planning and operational focus by highlighting regional revenue drivers and emerging risks.

---

## Connect

For collaboration, inquiries, or to discuss any of the projects:  
[Connect on LinkedIn](https://www.linkedin.com/in/serdaregeenderer/)

## License

This work is dual-licensed under **GPL-3.0-or-later** **and** **CC-BY-NC-4.0**.  
The content of this portfolio (dashboards, presentations, documentation) is licensed under the [**Creative Commons Attribution-NonCommercial 4.0 International**](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).  
All source code and Python scripts are licensed under the [**GNU General Public License v3.0**](https://spdx.org/licenses/GPL-3.0-or-later.html).

`SPDX-License-Identifier: GPL-3.0-or-later OR CC-BY-NC-4.0`
