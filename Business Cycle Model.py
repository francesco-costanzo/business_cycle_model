import pandas as pd
import pandas_datareader as pdr
from reportlab.pdfgen import canvas
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from reportlab.graphics.shapes import Drawing
from reportlab.lib.colors import red, blue, white, black, grey, green, greenyellow
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import quandl
import scipy.stats as sp
import seaborn as sns

from matplotlib.dates import date2num
from reportlab.lib.pagesizes import letter, landscape

location = '/Users/User/Desktop/Business Cycle'
dateToday = date.today()
print('Thinking...')

def rolling_zscore(economic_data):
    avg = economic_data.rolling(window=180).mean()
    std = economic_data.rolling(window=180).std()
    zscore = (economic_data - avg) / std
    function_res = sp.norm.cdf(zscore, 0)
    function_res = pd.DataFrame(function_res)
    function_res = 100 * (1 + ((2 * function_res) - 1))
    normalized = function_res.ewm(adjust=False, alpha=0.1, min_periods=12).mean()
    normalized.index = economic_data.index
    normalized = normalized.dropna()
    return normalized

def multi_period_return(return_stream):
    return np.prod(1 + return_stream) - 1 

def end_of_month(dates):
    delta_m = relativedelta(months=1)
    delta_d = timedelta(days=1)
    next_month = dates + delta_m
    end_of_month = date(next_month.year, next_month.month, 1) - delta_d
    return end_of_month

def fix_dates(series, missing_months):
    dates = series.index
    delta_d = timedelta(days=1)
    eomonth = dates - delta_d
    eolastmonth = dateToday - relativedelta(months=1)
    eolastmonth = end_of_month(eolastmonth)
    eo2month = dateToday - relativedelta(months=2)
    eo2month = end_of_month(eo2month)
    eo3month = dateToday - relativedelta(months=3)
    eo3month = end_of_month(eo3month)   
    if missing_months == 1:
        eomonth = list(eomonth)
        eomonth.append(eolastmonth)
        eomonth = pd.Series(eomonth)
    elif missing_months == 2:
        eomonth = list(eomonth)
        eomonth.append(eo2month)
        eomonth.append(eolastmonth)
        eomonth = pd.Series(eomonth)
    elif missing_months == 3:
        eomonth = list(eomonth)
        eomonth.append(eo3month)
        eomonth.append(eo2month)
        eomonth.append(eolastmonth)
        eomonth = pd.Series(eomonth)
    
    series.index = eomonth[len(eomonth) - len(series):]
    return series

quandl.ApiConfig.api_key = 'YFu19-LxqbmfKpsKkVAy'

# =============================================================================
#                          Composite Leading Indicator
# =============================================================================

eolastmonth = dateToday - relativedelta(months=1)
eolastmonth = end_of_month(eolastmonth)
eo2month = dateToday - relativedelta(months=2)
eo2month = end_of_month(eo2month)
end_date = eolastmonth

claims = pdr.get_data_fred('IC4WSA', start='1967-01-01', end=eolastmonth)
claims = claims.resample('M').last()
normalized_claims = rolling_zscore(-claims)
normalized_claims.columns = ['Initial Unemployment Claims']

man_hours = pdr.get_data_fred('AWHMAN', start='1939-01-01', end=end_date)
normalized_man_hours = rolling_zscore(man_hours)
normalized_man_hours = fix_dates(normalized_man_hours, 3)
normalized_man_hours.columns = ['Manufacturing Hours Worked']

ism = quandl.get('ISM/MAN_NEWORDERS', start_date='1939-01-01', end_date=end_date)['Index']
ism = fix_dates(ism, 1)
normalized_ism = rolling_zscore(ism)
normalized_ism.columns = ['ISM New Orders PMI']

house = pdr.get_data_fred('HOUST', start='1959-01-01', end=end_date)
house = fix_dates(house, 2)
normalized_house = rolling_zscore(house)
normalized_house.columns = ['New Housing Starts']

us10yr = pdr.get_data_fred('DGS10', start='1962-01-02', end=eolastmonth)
us10yr = us10yr.resample('M').last()
fed_funds = pdr.get_data_fred('DFF', start='1962-01-02', end=eolastmonth)
fed_funds = fed_funds.resample('M').last()
slope = us10yr - fed_funds.values
normalized_slope = rolling_zscore(slope)
normalized_slope.columns = ['Slope of Yield Curve']

m2 = pdr.get_data_fred('M2SL', start='1959-01-01', end=end_date)
m2 = m2.pct_change()
m2 = m2.rolling(window=6).apply(multi_period_return, raw=False).dropna()
m2 = fix_dates(m2, 2)
normalized_m2 = rolling_zscore(m2)
normalized_m2.columns = ['M2 Money Supply']

comp_lead_ind = (normalized_claims[(len(normalized_claims) - len(normalized_claims)):] * 0.1 + 
                 normalized_man_hours[(len(normalized_man_hours) - len(normalized_claims)):].values * 0.1 + 
                 normalized_ism[(len(normalized_ism) - len(normalized_claims)):].values * 0.2 + 
                 normalized_house[(len(normalized_house) - len(normalized_claims)):].values * 0.1 + 
                 normalized_slope[(len(normalized_slope) - len(normalized_claims)):].values * 0.25 + 
                 normalized_m2[(len(normalized_m2) - len(normalized_claims)):].values * 0.25)

comp_lead_ind.columns = ['Composite Leading Indicator']

# =============================================================================
#                         Composite Coinicident Indicator
# =============================================================================

nonfarm = pdr.get_data_fred('PAYEMS', start='1939-01-01', end=end_date)
nonfarm = fix_dates(nonfarm, 1)
nonfarm_yoy = nonfarm.pct_change()
nonfarm_yoy = nonfarm_yoy.rolling(12).apply(multi_period_return, raw=False)
normalized_nonfarm = rolling_zscore(nonfarm_yoy)
normalized_nonfarm.columns = ['Nonfarm Payrolls']

pers_inc = pdr.get_data_fred('DSPIC96', start='1959-01-01', end=end_date)
pers_inc = fix_dates(pers_inc, 2)
pers_inc = pers_inc.pct_change()
pers_inc = pers_inc.rolling(12).apply(multi_period_return, raw=False).dropna()
normalized_pers_inc = rolling_zscore(pers_inc)
normalized_pers_inc.columns = ['Personal Income']

ind_prod = pdr.get_data_fred('INDPRO', start='1939-01-01', end=end_date)
ind_prod = ind_prod.pct_change()
ind_prod = ind_prod.rolling(12).apply(multi_period_return, raw=False)
ind_prod = ind_prod.dropna()
ind_prod = fix_dates(ind_prod, 2)
normalized_ind_prod = rolling_zscore(ind_prod)
normalized_ind_prod.columns = ['Industrial Production']

#FRED has 2 data sets for manufacturing and trade sales, combine in 1967
man_sales1 = pdr.get_data_fred('M0602BUSM144NNBR', start='1939-01-01', end='1967-12-31')
man_sales1 = fix_dates(man_sales1, 0)
man_sales1 = man_sales1.pct_change()
man_sales1 = man_sales1.rolling(12).apply(multi_period_return, raw=False)
man_sales1 = man_sales1.dropna()
man_sales1.columns = ['Manufacturing Sales']
man_sales2 = pdr.get_data_fred('CMRMTSPL', start='1967-01-01', end=end_date)
man_sales2 = fix_dates(man_sales2, 3)
man_sales2 = man_sales2.pct_change()
man_sales2 = man_sales2.rolling(12).apply(multi_period_return, raw=False)
man_sales2 = man_sales2.dropna()
man_sales2.columns = ['Manufacturing Sales']

man_sales = pd.concat([man_sales1, man_sales2])
normalized_man_sales = rolling_zscore(man_sales)
normalized_man_sales.columns = ['Manufacturing Sales']

comp_coin_ind = (0.10 * normalized_pers_inc +
                 0.25 * normalized_nonfarm.values[len(normalized_nonfarm) - len(normalized_pers_inc):] + 
                 0.35 * normalized_ind_prod.values[len(normalized_ind_prod) - len(normalized_pers_inc):] +
                 0.30 * normalized_man_sales.values[len(normalized_man_sales) - len(normalized_pers_inc):]
        )

comp_coin_ind.columns = ['Composite Coincident Indicator']

# =============================================================================
#                       Composite Lagging Indicator
# =============================================================================

dur_unemp = pdr.get_data_fred('LNU03008275', start='1948-01-01', end=end_date)
dur_unemp = fix_dates(dur_unemp, 1)
normalized_dur_unemp = rolling_zscore(-dur_unemp)
normalized_dur_unemp.columns = ['Average Duration Unemployment']

prime_rate = pdr.get_data_fred('MPRIME', start='1949-01-01', end=end_date)
prime_rate = fix_dates(prime_rate, 1)
prime_rate = prime_rate.pct_change()
prime_rate = prime_rate.rolling(12).apply(multi_period_return, raw=False)
normalized_prime_rate = rolling_zscore(prime_rate)
normalized_prime_rate.columns = ['Average Prime Rate']

cons_credit = pdr.get_data_fred('TOTALSL', start='1943-01-01', end=end_date)
cons_credit = fix_dates(cons_credit, 3)
cons_credit = cons_credit.pct_change()
cons_credit = cons_credit.rolling(12).apply(multi_period_return, raw=False)
normalized_cons_credit = rolling_zscore(cons_credit)
normalized_cons_credit.columns = ['Consumer Credit Outstanding']

cpi_serv = pdr.get_data_fred('CUSR0000SAS', start='1956-01-01', end=end_date)
cpi_serv = fix_dates(cpi_serv, 1)
cpi_serv = cpi_serv.pct_change()
cpi_serv = cpi_serv.rolling(12).apply(multi_period_return, raw=False).dropna()
normalized_cpi = rolling_zscore(-cpi_serv)
normalized_cpi.columns = ['CPI Services']

comp_lag_ind = (0.25 * normalized_cpi + 
                0.25 * normalized_prime_rate.values[len(normalized_prime_rate) - len(normalized_cpi):] + 
                0.25 * normalized_cons_credit.values[len(normalized_cons_credit) - len(normalized_cpi):] + 
                0.25 * normalized_cpi.values[len(normalized_cpi) - len(normalized_cpi):]
                )
comp_lag_ind.columns = ['Composite Lagging Indicator']

# =============================================================================
#                   Composite Business Cycle Indicator
# =============================================================================
us_comp_bc = (0.5 * comp_lead_ind + 
              1/3 * comp_coin_ind[len(comp_coin_ind) - len(comp_lead_ind):].values + 
              1/6 * comp_lag_ind[len(comp_lag_ind) - len(comp_lead_ind):].values)

bc_3m = (us_comp_bc.rolling(window=3).mean() - 
         us_comp_bc.rolling(window=12).mean()) / us_comp_bc.rolling(window=12).mean()

bc_6m = us_comp_bc.pct_change()
bc_6m = bc_6m.rolling(window=6).apply(multi_period_return, raw=False)

bc_momo = 0.75 *  bc_3m + 0.25 * bc_6m

bus_cycle_signal = []

for i,j in zip(us_comp_bc.values, bc_momo.values):
    if i > 100 and j > 0:
        bus_cycle_signal.append(1)
    elif i > 100 and j < 0:
        bus_cycle_signal.append(2)
    elif i < 100 and j < 0:
        bus_cycle_signal.append(3)
    elif i < 100 and j > 0:
        bus_cycle_signal.append(4)
    else:
        bus_cycle_signal.append(np.NaN)

bc_signal = pd.Series(bus_cycle_signal, index=us_comp_bc.index[len(us_comp_bc) - len(bus_cycle_signal):])
bc_signal = bc_signal.dropna()
bc_signal = bc_signal.rename('Business_Cycle_Indicator')

ee = []
le = []
ec = []
lc = []
for phase in bc_signal:
    if phase == 1:
        ee.append(1)
        le.append(np.NaN)
        ec.append(np.NaN)
        lc.append(np.NaN)
    elif phase == 2:
        ee.append(np.NaN)
        le.append(1)
        ec.append(np.NaN)
        lc.append(np.NaN)
    elif phase == 3:
        ee.append(np.NaN)
        le.append(np.NaN)
        ec.append(1)
        lc.append(np.NaN)
    elif phase == 4:
        ee.append(np.NaN)
        le.append(np.NaN)
        ec.append(np.NaN)
        lc.append(1)
    else:
        pass

phases = pd.DataFrame({'Early_Expansion':ee, 'Late_Expansion':le,
                       'Early_Contraction':ec, 'Late_Contraction':lc},
                      index=bc_signal.index
    )

# =============================================================================
#                            Sector Returns
# =============================================================================
def get_sector_etf_rets(tickers, sector_names, start):
    data = []
    for etf, name in zip(tickers, sector_names):
        price = pdr.get_data_yahoo(etf, start=start, end=str(dateToday))['Adj Close']
        price = price.resample('M').last()
        rets = price.pct_change().dropna()
        rets = pd.DataFrame({name:rets})
        data.append(rets)
    sector_returns = data[0].join(data[1:])
    return sector_returns

tickers = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLU']
sector_names = ['CD', 'CS', 'EN', 'FN', 'HC', 'ID', 'IT', 'MA', 'UT']

sector_rets = get_sector_etf_rets(tickers, sector_names, '1998-12-16')
phase_rets = []

for phase in phases.columns:
    rets = get_sector_etf_rets(tickers, sector_names, '1998-12-16')
    for sector in sector_rets.columns:
        rets[sector] = sector_rets[sector] * phases[phase]
    rets = rets.dropna()
    ann_ret = np.prod(1 + rets) ** (1/(len(rets)/12)) - 1
    ann_stdev = np.std(rets) * np.sqrt(12)
    phase_ir = ann_ret / ann_stdev
    phase_ir = pd.DataFrame({phase:phase_ir})
    phase_rets.append(phase_ir)
    print('Still thinking...')
phase_rets = phase_rets[0].join(phase_rets[1:])

def ir_charts(phases_bc, top_bottom):
    fig = plt.figure(figsize=[11,9])
    for i, name in enumerate(phases_bc):
        i += 1
        ax = fig.add_subplot(2,2,i)
        ax.title.set_text(name)
        ax.tick_params(direction='in', length=8)
        colors = ['r' if i <= phase_rets[name].nsmallest(top_bottom).iloc[-1] 
          else 'g' if i >= phase_rets[name].nlargest(top_bottom).iloc[-1] 
          else 'grey' for i in phase_rets[name].values]
        phase_rets[name].plot(kind='bar', color=colors)
        ax.axhline(y=0, color='black', linewidth=1)
    return fig

charts = ir_charts(phase_rets.columns, 3)
plt.savefig(f'{location}/IR_Charts.png')        
spy = pdr.get_data_yahoo('SPY', start='1998-12-16', end=str(dateToday))['Adj Close']
spy = spy.resample('M').last()
spy = spy.pct_change().dropna()
spy_tr = spy / (np.std(spy) * np.sqrt(12) * 100)
# =============================================================================
#                            Sector Strategy
# =============================================================================
num_pos = 3
longs = [phase_rets[name].nlargest(num_pos).index.values for name in phase_rets.columns]
shorts = [phase_rets[name].nsmallest(num_pos).index.values for name in phase_rets.columns]

long_rets = [sector_rets[name] for name in longs]
long_rets = [(returns.sum(axis=1) * 1/num_pos) for returns in long_rets]
long_rets = pd.concat(long_rets, axis=1)
long_rets = long_rets.mul(phases.shift(1)[len(phases)-len(long_rets):].values)
long_rets = long_rets.sum(axis = 1)
long_rets_tr = long_rets / (np.std(long_rets) * np.sqrt(12) * 100)

short_rets = [sector_rets[name] for name in shorts]
short_rets = [(returns.sum(axis=1) * 1/num_pos) for returns in short_rets]
short_rets = pd.concat(short_rets, axis=1)
short_rets = short_rets.mul(phases.shift(1)[len(phases)-len(short_rets):].values)
short_rets = short_rets.sum(axis = 1)
short_rets_tr = short_rets / (np.std(short_rets) * np.sqrt(12) * 100)

long_short = long_rets - short_rets.values
long_short_tr = long_short / (np.sqrt(12) * np.std(long_short) * 100)

cum_long_ret = np.cumprod(1 + long_rets) - 1
cum_short_ret = np.cumprod(1 + short_rets) - 1
cum_long_short = np.cumprod(1 + long_short) - 1
cum_spy = np.cumprod(1 + spy) - 1

cum_long_ret_tr = np.cumprod(1 + long_rets_tr) - 1
cum_short_ret_tr = np.cumprod(1 + short_rets_tr) - 1
cum_long_short_tr = np.cumprod(1 + long_short_tr) - 1
cum_spy_tr = np.cumprod(1 + spy_tr) - 1

# =============================================================================
#                        Factor Rotation Strategy
# =============================================================================
'''Factor data drawn from Github with Bloomberg spreadsheet returns for 
periods 02/1994 - 02/2020. All other returns sourced from Yahoo Finance
ETFs'''

bb_factor_rets = pd.read_csv(
    '/Users/User/Desktop/Business Cycle/Factor_Returns.csv',
    index_col=0)
bb_end = str(bb_factor_rets.index[-1])
def get_factor_returns(tickers):
    factor_rets = []
    for etf in tickers:
        price = pdr.get_data_yahoo(etf, start=bb_end, end=str(eolastmonth))['Adj Close']
        rets = price.resample('M').last().pct_change().dropna()
        rets = pd.DataFrame({etf:rets})
        factor_rets.append(rets)
    output = factor_rets[0].join(factor_rets[1:], how='left')
    return output

f_rets = get_factor_returns(['VLUE','MTUM','IVW','SPLV','SHY','SPY'])
f_rets_comb = pd.DataFrame(np.concatenate((bb_factor_rets.values,
                                           f_rets), axis=0))
f_rets_comb.columns = bb_factor_rets.columns
f_rets_comb.index = pd.concat([bb_factor_rets, f_rets], axis=1).index
cum_spx_rets = np.cumprod(1 + (f_rets_comb['SPX Index'] / 
                               (np.std(f_rets_comb['SPX Index']) * np.sqrt(12) * 100))) - 1

phase_factor_rets = []
for phase in phases.columns:
    curr_phase = phases[phase][len(phases) - len(f_rets_comb):]
    curr_phase = pd.DataFrame([curr_phase] * 6).T
    rets = pd.DataFrame(f_rets_comb.values * curr_phase.values,
                        index=f_rets_comb.index, 
                        columns=f_rets_comb.columns)
    rets = rets.dropna()
    factor_rets = rets.drop('SPX Index', axis=1)
    spx_rets = rets['SPX Index']
    ann_ret = np.prod(1 + factor_rets) ** (1/(len(factor_rets)/12)) - 1
    ann_spx = np.prod(1 + spx_rets) ** (1/(len(spx_rets)/12)) - 1
    ann_stdev = np.std(factor_rets) * np.sqrt(12)
    phase_ir = (ann_ret - ann_spx) / ann_stdev
    phase_ir = pd.DataFrame({phase:phase_ir})
    phase_factor_rets.append(phase_ir)
phase_factor_rets = phase_factor_rets[0].join(phase_factor_rets[1:])
pivot = phase_factor_rets.T.to_numpy()

fig, ax = plt.subplots(figsize=[9.5,7])
im = ax.imshow(pivot, 'Blues')
im.set_clim(-0.5, 0.5)
plt.yticks(range(0,len(phase_factor_rets.columns),1))
plt.xticks(range(0,len(phase_factor_rets.index),1))
ax.set_yticklabels(phase_factor_rets.columns)
ax.set_xticklabels(['Value', 'Momentum', 'Growth', 'Low Volatility', 'Safe Haven'])
plt.setp(ax.get_xticklabels(), rotation=90, ha='right',rotation_mode='anchor')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Information Ratio', rotation=-90, va='bottom')

def annotate_heatmap(im, data=None, valfmt='{x:.2f}',
                     textcolors=['black', 'white'],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    kw = dict(horizontalalignment='center',
              verticalalignment='center')
    kw.update(textkw)
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


texts = annotate_heatmap(im, valfmt='{x:.3f}', threshold=0.25)
ax.add_patch(Rectangle((0.5,-0.5), 1, 1, fill=False, edgecolor='gold', lw=4))
ax.add_patch(Rectangle((1.5,2.5), 1, 1, fill=False, edgecolor='gold', lw=4))
ax.add_patch(Rectangle((2.5,0.5), 1, 1, fill=False, edgecolor='gold', lw=4))
ax.add_patch(Rectangle((3.5,1.5), 1, 1, fill=False, edgecolor='gold', lw=4))
plt.savefig(f'{location}/Heat_Map.png', dpi=300, bbox_inches='tight')

factor_signal = bc_signal[len(bc_signal)-len(f_rets_comb)-1:].shift(1).dropna()
f_strat_rets = []
for phase, mo, lv, sh, gr in zip(factor_signal, f_rets_comb['M2US000$ Index'],
                                 f_rets_comb['SP5LVIT Index'], f_rets_comb['LT01TRUU Index'],
                                 f_rets_comb['SPTRSGX Index']):
    if phase == 1:
        f_strat_rets.append(mo)
    elif phase == 2:
        f_strat_rets.append(lv)
    elif phase == 3:
        f_strat_rets.append(sh)
    elif phase == 4:
        f_strat_rets.append(gr)

f_strat_rets = pd.Series(f_strat_rets, index=f_rets_comb.index)
f_rets_tr = f_strat_rets / (np.std(f_strat_rets) * np.sqrt(12) * 100)
cum_f_rets = np.cumprod(1 + f_rets_tr) - 1

# =============================================================================
#                     Output 
# =============================================================================
pdf = canvas.Canvas(f'/Users/User/Desktop/Business Cycle/Business Cycle Monthly Report - {(dateToday - relativedelta(months=1)).strftime("%b")} {(dateToday- relativedelta(months=1)).year}.pdf')
pdf.setTitle('Business Cycle Monthly Report')

pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 770, 700, 150, fill=1)

pdf.setFillColor(white)
pdf.setFont('Helvetica', 48)
pdf.drawCentredString(300, 790, 'US Business Cycle Report')

pdf.setFillColor(grey, alpha=0.7)
pdf.setLineWidth(0)
pdf.rect(-10, 735, 620, 35, fill=1)

pdf.setFillColor(black)
pdf.setFont('Helvetica', 24)
lastmonth = (dateToday - relativedelta(months=1)).strftime('%B %Y')
pdf.drawString(50, 740, 'For the Month of ' + lastmonth)

def get_index_rets(tickers, names):
    index_rets = [pd.DataFrame({name:pdr.get_data_yahoo(ticker, start=str(dateToday - relativedelta(years=1)),
                                      end=str(end_date))['Adj Close']}) for name, ticker in zip(names, tickers)]
    index_rets = index_rets[0].join(index_rets[1:])
    index_rets = index_rets.pct_change().dropna()
    index_rets.iloc[0,:] = 0
    index_rets = np.cumprod(1 + index_rets) - 1
    return index_rets

index_rets = get_index_rets(['^IXIC', '^GSPC', '^RUT'],
                            ['NASDAQ Composite', 'S&P 500', 'Russell 2000'])
index_rets = index_rets * 100
plt.rcParams["font.family"] = "sans-serif"
fig, ax = plt.subplots(figsize=[8.5,4])
ax = index_rets['S&P 500'].plot(color='blue', label='S&P 500')
index_rets['NASDAQ Composite'].plot(color='dimgray', label='NASDAQ Composite', linestyle='dotted')
index_rets['Russell 2000'].plot(color='black', label='Russell 2000', linestyle='--')
ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
ax.axhline(y=0, color='black', linewidth=1)
ax.tick_params(direction='in', length=8)
ax.legend(edgecolor='black', loc='lower left')
x_axis = ax.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)
plt.savefig(f'{location}/Pic.png')
pdf.drawImage(f'{location}/Pic.png', 0, 450, 600, 285)
pdf.setFont('Helvetica', 14)
pdf.drawString(75, 710, 'US Equity Indices Trailing 12 Month Cumulative Returns')
pdf.setFont('Helvetica', 9)
pdf.drawString(75, 460, 'Source: Yahoo Finance, as of ' + end_date.strftime('%Y-%m-%d'))

pdf.setFont('Helvetica', 24)
pdf.drawString(50, 425, 'Monthly Summary')
pdf.line(47, 417, 550, 417)

pdf.setFont('Helvetica', 18)
pdf.drawString(100,390, 'Business Cycle Phase')
pdf.drawString(350,390, 'Target Asset Allocation')


current_phase = bus_cycle_signal[-1]
if current_phase == 1:
    pdf.setLineWidth(1)
    pdf.setFillColorRGB(0.1, 0.75, 0.35)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.setFillColorRGB(0.1, 0.75, 0.35, alpha=1)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 22)
    pdf.drawString(92.5, 332, 'Early Expansion')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(102, 247, 'Late Expansion')
    pdf.drawString(95, 162, 'Early Contraction')
    pdf.drawString(98, 77, 'Late Contraction')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(340, 345, f'Target Sectors: {", ".join(longs[0])}')
    pdf.drawString(341, 325, 'Target Factor: Momentum')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(341, 260, f'Target Sectors: {", ".join(longs[1])}')
    pdf.drawString(340, 240, 'Target Factor: Low Volatility')
    pdf.drawString(338, 175, f'Target Sectors: {", ".join(longs[2])}')
    pdf.drawString(345, 155, 'Target Factor: Safe Haven')
    pdf.drawString(343, 90, f'Target Sectors: {", ".join(longs[3])}')
    pdf.drawString(362, 70, 'Target Factor: Growth')
elif current_phase == 2:
    pdf.setLineWidth(1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.setFillColorRGB(0.6, 0.9, 0.4, alpha=1)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.setFillColorRGB(0.6, 0.9, 0.4, alpha=1)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 332, 'Early Expansion')
    pdf.setFont('Helvetica-Bold', 22)
    pdf.setFillColor(white)
    pdf.drawString(95, 247, 'Late Expansion')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 162, 'Early Contraction')
    pdf.drawString(98, 77, 'Late Contraction')
    pdf.setFont('Helvetica', 16)
    pdf.drawString(342, 345, f'Target Sectors: {", ".join(longs[0])}')
    pdf.drawString(344, 325, 'Target Factor: Momentum')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(337, 260, f'Target Sectors: {", ".join(longs[1])}')
    pdf.drawString(331, 240, 'Target Factor: Low Volatility')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(338, 175, f'Target Sectors: {", ".join(longs[2])}')
    pdf.drawString(345, 155, 'Target Factor: Safe Haven')
    pdf.drawString(343, 90, f'Target Sectors: {", ".join(longs[3])}')
    pdf.drawString(362, 70, 'Target Factor: Growth')
elif current_phase == 3:
    pdf.setLineWidth(1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.setFillColor(red, alpha=1)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.setFillColor(red, alpha=0.85)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 332, 'Early Expansion')
    pdf.drawString(102, 247, 'Late Expansion')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 22)
    pdf.drawString(85, 162, 'Early Contraction')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(98, 77, 'Late Contraction')
    pdf.setFont('Helvetica', 16)
    pdf.drawString(342, 345, f'Target Sectors: {", ".join(longs[0])}')
    pdf.drawString(344, 325, 'Target Factor: Momentum')
    pdf.drawString(341, 260, f'Target Sectors: {", ".join(longs[1])}')
    pdf.drawString(340, 240, 'Target Factor: Low Volatility')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(334, 175, f'Target Sectors: {", ".join(longs[2])}')
    pdf.drawString(338, 155, 'Target Factor: Safe Haven')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(343, 90, f'Target Sectors: {", ".join(longs[3])}')
    pdf.drawString(362, 70, 'Target Factor: Growth')
elif current_phase == 4:
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.setFillColorRGB(1,0.85,0, alpha=1.0)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.setFillColorRGB(1,0.85,0, alpha=1.0)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 332, 'Early Expansion')
    pdf.drawString(102, 247, 'Late Expansion')
    pdf.drawString(95, 162, 'Early Contraction')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 22)
    pdf.drawString(88, 77, 'Late Contraction')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(342, 345, f'Target Sectors: {", ".join(longs[0])}')
    pdf.drawString(344, 325, 'Target Factor: Momentum')
    pdf.drawString(341, 260, f'Target Sectors: {", ".join(longs[1])}')
    pdf.drawString(340, 240, 'Target Factor: Low Volatility')
    pdf.drawString(338, 175, f'Target Sectors: {", ".join(longs[2])}')
    pdf.drawString(345, 155, 'Target Factor: Safe Haven')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(336, 90, f'Target Sectors: {", ".join(longs[3])}')
    pdf.drawString(355, 70, 'Target Factor: Growth')
else:
    pass


# =============================================================================
#                      Page 1 - BC
# =============================================================================
pdf.showPage()

bc = pd.DataFrame({
    'BC': us_comp_bc.iloc[-36:,0],
    'Momo': bc_momo.iloc[-36:,0]
    }, index=us_comp_bc.index[-36:])

fig, ax = plt.subplots(figsize=[9.5,4])
ax.plot(us_comp_bc, color='blue')
ax.tick_params(direction='in', length=8)
ax.axvspan(datetime(1980,1,1), datetime(1980,7,31), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(1981,7,1)), date2num(datetime(1982,11,30)), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(1990,7,1)), date2num(datetime(1991,3,31)), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(2001,3,1)), date2num(datetime(2001,11,30)), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(2007,12,1)), date2num(datetime(2009,6,30)), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(2020,3,1)), date2num(datetime(2020,12,30)), color='grey', alpha=0.5)
ax.axhline(y=100, color='black', linewidth=1)
ax.set_ylim(25, 175)
plt.savefig(f'{location}/Bus_Cycle.png')
pdf.drawImage(f'{location}/Bus_Cycle.png', -40, 480, 700, 270)
            
fig, ax = plt.subplots(figsize=[7.5,5])
ax.plot(bc.Momo * 100, bc.BC, color='black')
ax.set_xlim(15, -15)
ax.set_ylim(60,140)
plt.axhline(y=100, color='black', linewidth=1)
plt.axvline(x=0, color='black', linewidth=1)
ax.get_xaxis().set_major_formatter(ticker.PercentFormatter())
ax.set_xlabel('Index Momentum')
ax.set_ylabel('Index Level')
ax.axvspan(0, 15, 0.5, 1, color='green', alpha=0.7)
ax.axvspan(-15, 0, 0.5, 1, color='green', alpha=0.4)
ax.axvspan(-15, 0, 0, 0.5, color='red', alpha=0.6)
ax.axvspan(0, 15, 0, 0.5, color='gold', alpha=0.6)
plt.annotate('{:%b-%y}'.format(pd.to_datetime(bc.index[0])), xy=(bc.Momo[1] * 100 + 2, bc.BC[1]-5))
plt.annotate('{:%b-%y}'.format(pd.to_datetime(bc.index[-1])), xy=(bc.Momo[-1]*100 + 3, bc.BC[-1]-5))
plt.savefig(f'{location}/BC_Map.png')
pdf.drawImage(f'{location}/BC_Map.png', 40, 80, 550, 350)

pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'Composite Business Cycle Indicator')
pdf.setFont('Helvetica', size=18)
pdf.drawString(50,740, 'Composite Business Cycle Indicator')
pdf.drawString(50,425, 'Business Cycle Compass')
pdf.setLineWidth(1)
pdf.line(45, 417, 550, 417)
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 479, 'Note: Shaded areas represent US recessions as indicated by NBER.')
pdf.drawString(47, 466, 'Reading greater than 100 indicates expansion, less than 100 indicates contraction.')
pdf.setFont('Helvetica', size=32)
pdf.setFillColor(black)
pdf.drawString(120, 360, '1')
pdf.drawString(500, 360, '2')
pdf.drawString(500, 125, '3')
pdf.drawString(120, 125, '4')
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 75, 'Note: Momentum calculated as 25% 6M Change, 75% 3M Average less 12M Average.')
pdf.drawString(47, 62, '1 = Early Expansion, 2 = Late Expansion, 3 = Early Contraction, 4 = Late Contraction')
pdf.line(45, 57, 550, 57)


# =============================================================================
#                          Sector Strategy
# =============================================================================
pdf.showPage()

pdf.setFillColor(black)
pdf.drawImage(f'{location}/IR_Charts.png', -25, 290, width=635, height=495)

pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'US Sector Rotation Strategy')
pdf.setFont('Helvetica', size=18)
pdf.drawString(40,750,'US Sectors Information Ratios by Phase of Business Cycle')
pdf.setLineWidth(1)
pdf.line(35, 745, 560, 745)

tr_1 = True

if tr_1 == True:
    fig, ax = plt.subplots(figsize=[8.5,4])
    ax = (cum_long_ret_tr * 100).plot(color='green', label='Long')
    (cum_short_ret_tr * 100).plot(color='red', label='Short')
    (cum_long_short_tr * 100).plot(color='blue', label='Long-Short')
    (cum_spy_tr * 100).plot(color='black', label='SPY')
    ax.legend(edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.tick_params(which='major', direction='in', length=8)
    ax.tick_params(which='minor', direction='in', length=4)
    ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    plt.savefig(f'{location}/SectorIRs.png')
    pdf.drawImage(f'{location}/SectorIRs.png', -10 , 3, 590, 290)
    pdf.setFontSize(10)
    pdf.drawString(45, 309, 'Note: Green bars denote sectors with long position, red short, and grey neutral.')
    pdf.drawString(45, 296, 'Sector returns based on SPDR Sector ETF returns from 1999-01-31 to ' + eolastmonth.strftime('%Y-%m-%d'))
    pdf.line(35, 291, 560, 291)
    pdf.setFontSize(18)
    pdf.drawString(45, 264, 'Sector Strategy Cumulative Returns, Risk Level 1%')
elif tr_1 == False:
    fig, ax = plt.subplots(figsize=[8.5,4])
    ax = (cum_long_ret * 100).plot(color='green', label='Long', linestyle='--')
    (cum_short_ret * 100).plot(color='red', label='Short', linestyle='--')
    (cum_long_short * 100).plot(color='blue', label='Long-Short')
    (cum_spy * 100).plot(color='black', label='SPY')
    ax.legend(edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.tick_params(which='major', direction='in', length=8)
    ax.tick_params(which='minor', direction='in', length=4)
    ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    plt.savefig(f'{location}/SectorIRs.png')
    pdf.drawImage(f'{location}/SectorIRs.png', -10 , 3, 590, 290)
    pdf.setFontSize(10)
    pdf.drawString(45, 309, 'Note: Green bars denote sectors with long position, red short, and grey neutral.')
    pdf.drawString(45, 296, 'Sector returns based on SPDR Sector ETF returns from 1999-01-31 to ' + eolastmonth.strftime('%Y-%m-%d'))
    pdf.line(35, 291, 560, 291)
    pdf.setFontSize(18)
    pdf.drawString(45, 264, 'Sector Strategy Cumulative Returns')

# =============================================================================
#                          Factor Strategy
# =============================================================================
pdf.showPage()
# counter = 0
# while counter <= 900:
#     if counter % 50 == 0:
#         pdf.line(0, counter, 25, counter)
#         pdf.drawString(30, counter, str(counter))
#     elif counter % 10 == 0:
#         pdf.line(0, counter, 10, counter)
#     counter += 1

pdf.drawImage(f'{location}/Heat_Map.png', 50, 415, width=540, height=305)
pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'US Factor Rotation Strategy')
pdf.setFont('Helvetica', size=18)
pdf.drawString(50,743,'US Factors Information Ratios by Phase of Business Cycle')
pdf.setLineWidth(1)
pdf.line(45, 736, 555, 736)
pdf.setFont('Helvetica', size=10)
pdf.drawString(50, 410, 'Note: Gold borders denote factors with long exposure during phase of Business Cycle.')
pdf.drawString(50, 397, 'Information Ratio calculated as factor excess return to S&P 500 Index, scaled by annualized volatility.')
pdf.drawString(50, 384, 'Returns sourced from period 1994-02-28 to ' + eolastmonth.strftime('%Y-%m-%d'))
pdf.line(45, 377, 555, 377)

pdf.setFontSize(18)

fig, ax = plt.subplots(figsize=[8.5,4])
ax = (cum_f_rets * 100).plot(color='blue', label='Factor Model')
(cum_spx_rets * 100).plot(color='black', linestyle='--', label='S&P 500 Index')
ax.legend(edgecolor='black')
ax.axhline(y=0, color='black', linewidth=1)
ax.tick_params(which='major', direction='in', length=8)
ax.tick_params(which='minor', direction='in', length=4)
ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
x_axis = ax.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)
plt.savefig(f'{location}/FactorReturns.png')
pdf.drawImage(f'{location}/FactorReturns.png', -10, 40, 620, 300)
pdf.drawString(50, 315, 'Factor Strategy Cumulative Returns, Risk Level 1%')


# =============================================================================
#                      Page 2 - Composite Leading Indicator
# =============================================================================
pdf.showPage()

comp_lead_indicators = [normalized_man_hours, comp_lead_ind,  
                        normalized_ism, normalized_claims,
                        normalized_house, normalized_slope,
                        normalized_m2]
comp_lead_indicators = comp_lead_indicators[0].join(comp_lead_indicators[1:], 
                                                    how='left')
comp_lead_indicators = comp_lead_indicators[['Composite Leading Indicator',
                         'M2 Money Supply',
                         'Slope of Yield Curve',
                         'New Housing Starts',
                         'ISM New Orders PMI',
                         'Manufacturing Hours Worked',
                         'Initial Unemployment Claims']]

pdf.setFont('Helvetica', 12)
pdf.setLineWidth(1)

def plot_indicator_charts(indicators, figsize, pic_num):
    fig, ax = plt.subplots(nrows=len(indicators.columns), figsize=figsize, sharex=True, 
                       gridspec_kw={'hspace':0})
    for i, name in enumerate(indicators):
        if i % 2 == 0:
            ax[i].plot(indicators[name], color='blue')
            ax[i].axvspan(datetime(1969,12,1), datetime(1970,11,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1973,11,1), datetime(1975,3,31), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1980,1,1), datetime(1980,7,31), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1981,7,1), datetime(1982,11,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1990,7,1), datetime(1991,3,31), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(2001,3,1), datetime(2001,11,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(2007,12,1), datetime(2009,6,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(2020,3,1), datetime(2020,12,30), color='grey', alpha=0.5)
            ax[i].axhline(y=100, color='black', linewidth=1)
            ax[i].legend([name], edgecolor='black', loc='lower left')
            ax[i].set_ylim(0, 200)
            ax[i].tick_params(direction='in', length=8)
            plt.savefig(f'{location}/Pic' + str(pic_num) +'.png')
        else:
            ax[i].plot(indicators[name], color='blue')
            ax[i].axvspan(datetime(1969,12,1), datetime(1970,11,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1973,11,1), datetime(1975,3,31), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1980,1,1), datetime(1980,7,31), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1981,7,1), datetime(1982,11,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(1990,7,1), datetime(1991,3,31), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(2001,3,1), datetime(2001,11,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(2007,12,1), datetime(2009,6,30), color='grey', alpha=0.5)
            ax[i].axvspan(datetime(2020,3,1), datetime(2020,12,30), color='grey', alpha=0.5)
            ax[i].axhline(y=100, color='black', linewidth=1)
            ax[i].legend([name], edgecolor='black', loc='lower left')
            ax[i].set_ylim(0, 200)
            ax[i].tick_params(direction='in', length=8)
            ax[i].yaxis.tick_right()
            plt.savefig(f'{location}/Pic' + str(pic_num) +'.png')
    return fig




plot_indicator_charts(comp_lead_indicators[200:], [8.5,12], 2)
pdf.drawImage(f'{location}/Pic2.png', -10, 10, 600, 850)
pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'Composite Leading Indicators')
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 65, 'Note: Shaded areas represent US recessions as indicated by NBER.')
pdf.drawString(47, 52, 'Reading greater than 100 indicates expansion, less than 100 indicates contraction.')

# =============================================================================
#                      Page 3 - Composite Coincident Indicator
# =============================================================================
pdf.showPage()
pdf.setLineWidth(1)
counter = 0
while counter <= 900:
    if counter % 50 == 0:
        pdf.line(0, counter, 25, counter)
        pdf.drawString(30, counter, str(counter))
    elif counter % 10 == 0:
        pdf.line(0, counter, 10, counter)
    counter += 1

comp_coin_indicators = [normalized_nonfarm, 
                        comp_coin_ind, normalized_ind_prod,
                        normalized_pers_inc, normalized_man_sales]

comp_coin_indicators = comp_coin_indicators[0].join(comp_coin_indicators[1:], how='left')
comp_coin_indicators = comp_coin_indicators[['Composite Coincident Indicator',
                                             'Nonfarm Payrolls',
                                             'Manufacturing Sales',
                                             'Industrial Production',
                                             'Personal Income']]

plot_indicator_charts(comp_coin_indicators[200:], [8.5,12], 3)
pdf.drawImage(f'{location}/Pic3.png', -10, 10, 600, 850)
pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'Composite Coincident Indicators')
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 65, 'Note: Shaded areas represent US recessions as indicated by NBER.')
pdf.drawString(47, 52, 'Reading greater than 100 indicates expansion, less than 100 indicates contraction.')

# =============================================================================
#                      Page 4 - Composite Lagging Indicator
# =============================================================================
pdf.showPage()

comp_lag_indicators = [normalized_cons_credit,
                       comp_lag_ind, normalized_dur_unemp,
                       normalized_prime_rate, 
                       normalized_cpi]

comp_lag_indicators = comp_lag_indicators[0].join(comp_lag_indicators[1:], how='left')
comp_lag_indicators = comp_lag_indicators[[
    'Composite Lagging Indicator',
    'Consumer Credit Outstanding',
    'Average Prime Rate',
    'Average Duration Unemployment',
    'CPI Services']]

plot_indicator_charts(comp_lag_indicators[150:], [8.5,12], 4)
pdf.drawImage(f'{location}/Pic4.png', -10, 10, 600, 850)
pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'Composite Lagging Indicators')
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 65, 'Note: Shaded areas represent US recessions as indicated by NBER.')
pdf.drawString(47, 52, 'Reading greater than 100 indicates expansion, less than 100 indicates contraction.')

pdf.save()
print('Done')