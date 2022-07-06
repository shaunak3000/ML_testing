# ML_testing
learning different ML approaches

##################################
# Shock_data.xlsx and shock_classify
This xlsx file contains data of operating cash flow shocks on publicly traded US firms
Shocks are identified using rolling volatility of 24 quarters of op_cash_income
If the shock happens due to increase in op_cash_income, it is dropped
If the shock coincides with an increase in firm performance it is dropped

I am using structured data classification using Keras
numerical features are normalized and categorical features are vectorized
