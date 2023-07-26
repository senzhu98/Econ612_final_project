%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Final Project%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
%% Note: final version 

%% read data
cd '/Users/ezreal/Library/CloudStorage/OneDrive-DukeUniversity/Duke 2023/Econ_612/Final Project'

url = 'https://fred.stlouisfed.org/';
c = fred(url);
series_cpi_air = 'CUSR0000SETG01';
data_cpi_air = fetch(c,series_cpi_air);
cpi_air_data = data_cpi_air.Data;
date_cpi_air = cpi_air_data(:,1);
cpi_air = cpi_air_data(:,2);
date_cpi_air = datetime(date_cpi_air,'ConvertFrom','datenum');

series_ppi = 'PPIACO';
data_ppi = fetch(c,series_ppi);
ppi_data = data_ppi.Data;
date_ppi = ppi_data(:,1);
ppi = ppi_data(:,2);
date_ppi = datetime(date_ppi,'ConvertFrom','datenum');

series_hs = 'HOUST';
data_hs = fetch(c,series_hs);
hs_data = data_hs.Data;
date_hs = hs_data(:,1);
hs = hs_data(:,2);
date_hs = datetime(date_hs,'ConvertFrom','datenum');

series_cs = 'UMCSENT'; % consumer sentiment
data_cs = fetch(c,series_cs);
cs_data = data_cs.Data;
date_cs = cs_data(:,1);
cs = cs_data(:,2);
date_cs = datetime(date_cs,'ConvertFrom','datenum');
plot(date_cs,cs)


%% data inspection
% time series plot
myplot1 = figure(1);
plot(date_cpi_air,cpi_air)
title("CPI: Airline fares in U.S. city average")
ylabel("Level")
xlabel("Time")
saveas(myplot1,'original_time_series_from_BLS.png')

%% preliminaries
% check unit root
adftest(cpi_air,Model="ARD",lags=12)
[h,pValue,stat,cValue] = adftest(cpi_air, Model="ARD", Alpha=[0.05 0.1], lags=12);

% Conclusion: fail to reject the null hypothesis that levels of cpi for airline fares
% has a unit root. so need to forecast growth rates instead of levels
% (include lagged dependent variable in regression)

%% data transformation (from levels to growth rate)
% for cpi: airline fares
cpi_air_lag = lagmatrix(cpi_air,1);
diff = cpi_air-cpi_air_lag;

cpi_air_gr = zeros(410,1);
for i = 1:410
    cpi_air_gr(i,1) = (diff(i))/cpi_air_lag(i);
end
myplot(2) = figure(2);
plot(date_cpi_air,cpi_air_gr);
xlabel("Time")
ylabel("Growth Rate")
title("CPI: Airline fares in U.S. city average")
saveas(myplot(2),"plot_growth_rate.png")

% change housing starts from levels to growth rate
hs_lag = lagmatrix(hs,1);
diff_hs = hs-hs_lag;

hs_gr = zeros(770,1);
for i = 1:770
    hs_gr(i,1) = (diff_hs(i))/hs_lag(i);
end
est_hs = (date_hs >= datetime(1989,1,1));
date_hs = date_hs(est_hs);
hs_gr = hs_gr(est_hs);

% DeSeason PPI
est_ppi = (date_ppi >= datetime(1989,1,1));
date_ppi = date_ppi(est_ppi);
ppi = ppi(est_ppi);
mo_ppi = dummyvar(month(date_ppi));
mdl2 = fitlm(mo_ppi,ppi,'Intercept',false);
res_ppi = mdl2.Residuals.Raw;
plot(date_ppi,res_ppi)
mu_ppi = mean(ppi);
T_ppi = length(date_ppi);
sadj_ppi = mu_ppi * ones(T_ppi,1) + res_ppi;

% transform PPI from levels to growth rates
sadj_ppi_lag = lagmatrix(sadj_ppi,1);
diff_ppi = sadj_ppi-sadj_ppi_lag;
ppi_gr = zeros(410,1);
for i = 1:410
    ppi_gr(i,1) = (diff_ppi(i))/sadj_ppi_lag(i);
end

% DeSeason consumer sentiment (UofM)
est_cs = (date_cs >= datetime(1989,1,1));
date_cs = date_cs(est_cs);
cs = cs(est_cs);
mo_cs = dummyvar(month(date_cs));
mdl_cs = fitlm(mo_cs,cs,'Intercept',false);
res_cs = mdl_cs.Residuals.Raw;
plot(date_cs,res_cs)
mu_cs = mean(cs);
T_cs = length(date_cs);
sadj_cs = mu_cs * ones(T_cs,1) + res_cs;

% transform consumer sentiment from levels to growth rates
sadj_cs_lag = lagmatrix(sadj_cs,1);
diff_cs = sadj_cs-sadj_cs_lag;
cs_gr = zeros(410,1);
for i = 1:410
    cs_gr(i,1) = (diff_cs(i))/sadj_cs_lag(i);
end







%% models specification and causality test
% Baseline AR(12) model
mdl1 = fitlm(lagmatrix(cpi_air_gr,1:12),cpi_air_gr);
fitted_mdl1 = predict(mdl1,[lagmatrix(cpi_air_gr,1:12)]);
myplot(2)=figure(2);
plot(date_cpi_air,cpi_air_gr,date_cpi_air,fitted_mdl1)
xlabel("Time")
ylabel("Growth Rate")
title("CPI: Airline fares in U.S. city average")
saveas(myplot(2),"plot_baseline_model.png")

% R-square = 0.273, not very good


% different leading indicators (ppi + housing starts + consumer sentiment)
% first check causality between cpi and ppi
mdl_cpi_ppi = fitlm([lagmatrix(ppi_gr,1:12) lagmatrix(cpi_air_gr,1:12)],cpi_air_gr);
[EstCov,se,coeff] = hac(mdl_cpi_ppi,'display','off',Type='HC');
R = zeros(12,25); R(1:12,2:13)=eye(12);
[h,pValue,stat,cValue]=waldtest(coeff(2:13,1),R,EstCov);

% h=1 and p-value is small : we reject the null hypothesis of non-causality

% then check causality between cpi and housing starts
mdl_cpi_hs = fitlm([lagmatrix(hs_gr,1:12) lagmatrix(cpi_air_gr,1:12)],cpi_air_gr);
[EstCov,se,coeff] = hac(mdl_cpi_hs,'display','off',Type='HC');
R = zeros(12,25); R(1:12,2:13)=eye(12);
[h,pValue,stat,cValue]=waldtest(coeff(2:13,1),R,EstCov);

% h = 0 and p-value = 0.7910 is large, so we fail to reject the null
% hypothesis that housing starts growth rate does not predictively cause
% cpi_air. unclear if grwoth rate of housing starts help predict cpi:
% airline fares

% then check causality between cpi and consumer sentiment
mdl_cpi_cs = fitlm([lagmatrix(cs_gr,1:12) lagmatrix(cpi_air_gr,1:12)],cpi_air_gr);
[EstCov,se,coeff] = hac(mdl_cpi_cs,'display','off',Type='HC');
R = zeros(12,25); R(1:12,2:13)=eye(12);
[h,pValue,stat,cValue]=waldtest(coeff(2:13,1),R,EstCov);
% h = 1 and p-value = 0.0213, we reject the null hypothesis of non-causality



%% model selection using AIC
% restric sample period from 1990m1 to 2023m2
est = (date_cpi_air >= datetime(1990,1,1));
date_cpi_air_new = date_cpi_air(est);

cpi_air_gr_new = cpi_air_gr(est);
ppi_gr_new = ppi_gr(est);
cs_gr_new = cs_gr(est);

% baseline model:
% Estimate AR(1-12) omitting first 12 observations
for i = 1:12
    mdl_ar = fitlm(lagmatrix(cpi_air_gr_new,1:i),cpi_air_gr_new);
    ar_AIC(i,1) = mdl_ar.ModelCriterion.AIC;
end
min(ar_AIC)
% smallest AIC is -1839.4 where p = 2. AR(2)


% combined model:
% 1,...,12 lags of cpi_air
% 1,..,6 lags of ppi
% 1,...,6 lags of consumer sentiment

% first check model including cpi_air and ppi
for i = 1:12
    mdl_cpi_ppi1 = fitlm([lagmatrix(cpi_air_gr_new,1:i) ppi_gr_new],cpi_air_gr_new);
    cpi_ppi_AIC(i,1) = mdl_cpi_ppi1.ModelCriterion.AIC;
end
for i = 1:12
    mdl_cpi_ppi2 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(ppi_gr_new,1)],cpi_air_gr_new);
    cpi_ppi_AIC(i,2) = mdl_cpi_ppi2.ModelCriterion.AIC;
end
for i = 1:12
    mdl_cpi_ppi_3 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(ppi_gr_new,2)],cpi_air_gr_new);
    cpi_ppi_AIC(i,3) = mdl_cpi_ppi_3.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_ppi4 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(ppi_gr_new,3)],cpi_air_gr_new);
    cpi_ppi_AIC(i,4) = mdl_cpi_ppi4.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_ppi5 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(ppi_gr_new,4)],cpi_air_gr_new);
    cpi_ppi_AIC(i,5) = mdl_cpi_ppi5.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_ppi6 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(ppi_gr_new,5)],cpi_air_gr_new);
    cpi_ppi_AIC(i,6) = mdl_cpi_ppi6.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_ppi7 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(ppi_gr_new,6)],cpi_air_gr_new);
    cpi_ppi_AIC(i,7) = mdl_cpi_ppi7.ModelCriterion.AIC;
end

min(cpi_ppi_AIC)
% smallest AIC is -1864.2 where p = 2, and q = 0


% then check model including cpi_air and consumer sentiment
for i = 1:12
    mdl_cpi_cs1 = fitlm([lagmatrix(cpi_air_gr_new,1:i) cs_gr_new],cpi_air_gr_new);
    cpi_cs_AIC(i,1) = mdl_cpi_cs1.ModelCriterion.AIC;
end
for i = 1:12
    mdl_cpi_cs2 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(cs_gr_new,1)],cpi_air_gr_new);
    cpi_cs_AIC(i,2) = mdl_cpi_cs2.ModelCriterion.AIC;
end
for i = 1:12
    mdl_cpi_cs3 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(cs_gr_new,2)],cpi_air_gr_new);
    cpi_cs_AIC(i,3) = mdl_cpi_cs3.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_cs4 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(cs_gr_new,3)],cpi_air_gr_new);
    cpi_cs_AIC(i,4) = mdl_cpi_cs4.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_cs5 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(cs_gr_new,4)],cpi_air_gr_new);
    cpi_cs_AIC(i,5) = mdl_cpi_cs5.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_cs6 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(cs_gr_new,5)],cpi_air_gr_new);
    cpi_cs_AIC(i,6) = mdl_cpi_cs6.ModelCriterion.AIC;
end

for i = 1:12
    mdl_cpi_cs7 = fitlm([lagmatrix(cpi_air_gr_new,1:i) lagmatrix(cs_gr_new,6)],cpi_air_gr_new);
    cpi_cs_AIC(i,7) = mdl_cpi_cs7.ModelCriterion.AIC;
end

min(cpi_cs_AIC)
% smallest AIC is -1837.7 where p = 2, and s = 1




% check AIC(p,q,s) = AIC(2,0,1)
combined_mdl = fitlm([lagmatrix(cpi_air_gr_new,1:2) ppi_gr_new cs_gr_new],cpi_air_gr_new);
combined_mdl.ModelCriterion.AIC
% AIC for combined model is -1863: not as small as -1864.2 where p = 2, and q = 0




% Conclusion: smallest AIC is is -1864.2 where p = 2, and q = 0
mdl_saic = fitlm([lagmatrix(cpi_air_gr_new,1:2) ppi_gr_new],cpi_air_gr_new);
fitted_mdl_saic= predict(mdl_saic,[lagmatrix(cpi_air_gr_new,1:2) ppi_gr_new]);
myplot(3)=figure(3);
plot(date_cpi_air_new,cpi_air_gr_new,date_cpi_air_new,fitted_mdl_saic)
xlabel("Time")
ylabel("Growth Rate")
title("CPI: Airline fares in U.S. city average")
saveas(myplot(3),"plot_smallest_aic_model.png")

% residual plot
myplot(4)=figure(4);
plot(date_cpi_air_new,-cpi_air_gr_new-fitted_mdl_saic)
xlabel("Time")
title('Rsiduals') % more like white noise
saveas(myplot(4),"residual_plot.png")


%% 12-month forecast
% 12-month ahead regression
mdl_forecast = fitlm([lagmatrix(cpi_air_gr_new,12:13) ppi_gr_new],cpi_air_gr_new);

% point and interval forecast for growth rate of cpi: airline fares (Direct method)
cpi_air_gr_new_f=zeros(12,1); cpi_air_gr_new_fi=zeros(12,2);
Tcpi = length(cpi_air_gr_new);
Tppi = length(ppi_gr_new);

for h=1:12
    mdl_p = fitlm([lagmatrix(cpi_air_gr_new,h:h+1) ppi_gr_new],cpi_air_gr_new);
    [cpi_air_gr_new_p,cpi_air_gr_new_pi] = predict(mdl_p,[fliplr(cpi_air_gr_new(Tcpi-1:Tcpi)') flip(ppi_gr_new(Tppi)')],'Prediction','observation','Alpha',0.1); 
    cpi_air_gr_newf(h,1) = cpi_air_gr_new_p;
    cpi_air_gr_new_fi(h,:) = cpi_air_gr_new_pi;
end

cpi_air_gr_newf
cpi_air_gr_new_fi

myplot(5) = figure(5);
plot(date_cpi_air_new(date_cpi_air_new>= datetime(2013,1,1)),cpi_air_gr_new(date_cpi_air_new>= datetime(2013,1,1)));
hold on;
plot(date_cpi_air_new(Tcpi)+calmonths(1:12),cpi_air_gr_newf,'LineWidth',1.5);
hold on;
plot(date_cpi_air_new(Tcpi)+calmonths(1:12),cpi_air_gr_new_fi,'LineWidth',1.5);
legend('cpi:airline fares growth rate','Direct forecast','Direct lower forecast interval','Direct upper forecast interval');
saveas(myplot(5),'cpi_growth_rate_forecast.png')

% return growth rate to levels
cpi_air_newf = zeros(12,1);
cpi_air_newf(1,1) = cpi_air_gr_newf(1,1)*cpi_air(end)+cpi_air(end);

cpi_air_new_fl = zeros(12,1);
cpi_air_new_fu = zeros(12,1);
cpi_air_new_fl(1,1) = cpi_air_gr_new_fi(1,1)*cpi_air(end)+cpi_air(end);
cpi_air_new_fu(1,1) = cpi_air_gr_new_fi(1,2)*cpi_air(end)+cpi_air(end);

% point forecast
for i = 2:12
    cpi_air_newf(i,1) = cpi_air_gr_newf(i,1)*cpi_air_newf(i-1,1)+cpi_air_newf(i-1,1);
end
% interval forecast
for i = 2:12
    cpi_air_new_fl(i,1) = cpi_air_gr_new_fi(i,1)*cpi_air_new_fl(i-1,1)+cpi_air_new_fl(i-1,1);
    cpi_air_new_fu(i,1) = cpi_air_gr_new_fi(i,2)*cpi_air_new_fu(i-1,1)+cpi_air_new_fu(i-1,1);
end

cpi_air_new_fi = cat(2,cpi_air_new_fl,cpi_air_new_fu);

% time series plot including 12 future forecasts
myplot(6) = figure(6);
plot(date_cpi_air(date_cpi_air>= datetime(2013,1,1)),cpi_air(date_cpi_air>= datetime(2013,1,1)));
hold on;
plot(date_cpi_air(length(date_cpi_air))+calmonths(1:12),cpi_air_newf,'LineWidth',1.5);
hold on;
plot(date_cpi_air(length(date_cpi_air))+calmonths(1:12),cpi_air_new_fi,'LineWidth',1.5);
hold on;
legend('cpi:airline fares levels','Direct forecast','Direct lower forecast interval','Direct upper forecast interval');
saveas(myplot(6),'cpi_air_levels_forecast.png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%The end%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

































