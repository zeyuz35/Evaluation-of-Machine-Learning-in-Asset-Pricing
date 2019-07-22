The data and code in this zip file may be used for non-commercial purposes free of charge. It is provided as is, without any guarantee of correctness. Please reference the relevant paper for further construction details and proper use. Email me if you have any issues with the data or code.

---------------
Please cite as:
---------------

He, Zhiguo, Bryan Kelly, and Asaf Manela, Intermediary Asset Pricing: New Evidence from Many Asset Classes, Journal of Financial Economics, 2017, Vol 126, Issue 1, pp. 1–35


-----------------------
File descriptions:
-----------------------

He_Kelly_Manela_Factors_And_Test_Assets.csv
-------------------------------------------
Quarterly time series of the He-Kelly-Manela intermediary capital measures for
the original sample period 1970Q1-2012Q4 as well as other factors and the net returns of test assets
used in the paper. When the original data was excess returns we added back the riskfree rate (rf). 
If you use the asset returns then consider citing the original sources (cited in our paper).


He_Kelly_Manela_Factors_And_Test_Assets_monthly.csv
---------------------------------------------------
Monthly time series similar to above file, but omitting some series which we
don't have in monthly format (e.g. aem_leverage_ratio).


He_Kelly_Manela_Factors_<daily/monthly/quarterly>.csv
-----------------------------------------------------
Updated quarterly, monthly, and daily time series of the He-Kelly-Manela
intermediary capital measures. Variables are the same as in the original dataset
for the period 1970Q1-2012Q4, but differ slightly starting in 2013Q1 because
new data is from Bloomberg, whereas original data was from CRSP-Compustat and
Datastream. The daily sample is entirely based on new data from Bloomberg.
Note that the last quarter's data is preliminary and often changes as more
firms report earnings during the subsequent quarter.

He_Kelly_Manela_XS_Tests.jl
-------------------------------------------
Replication code in Julia for main cross-sectional results.
Julia is free, fast, open-sourced, and endorsed by Tom Sargent (https://www.youtube.com/watch?v=KkKBwJkYgVk).
If you are fluent in matlab, this should be easy to read.
You can compare it with your favorite language at http://julialang.org
To run from command line simply cd to data and code folder and run:
	julia He_Kelly_Manela_XS_Tests.jl
It will create two csv files with cross-sectional test results at quarterly and monthly frequencies.


----------------------
Variable descriptions:
----------------------

intermediary_capital_ratio
--------------------------
The end of period ratio of total market cap to (total market cap + book assets - book equity)
of NY Fed primary dealers' publicly-traded holding companies.

intermediary_capital_risk_factor
--------------------------------
AR(1) innovations to the intermediary_capital_ratio scaled by lagged intermediary_capital_ratio.
Note: In extending the intermediary_capital_risk_factor, we retain the AR(1)
coefficients used in the paper.

intermediary_value_weighted_investment_return
---------------------------------------------
The value-weighted investment return to a portfolio of NY Fed primary dealers'
publicly-traded holding companies. Unlike the intermediary_capital_risk_factor,
this portfolio is tradable, and performed similarly as a pricing factor.

intermediary_leverage_ratio_squared
-----------------------------------
This level variable, defined as (1/intermediary_capital_ratio)^2 was used
for preliminary predictive tests in the paper, as prescribed by the HK model.

