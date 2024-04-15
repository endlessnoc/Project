import pandas as pd

def drawdown(return_series: pd.Series):
    """
    Takes a times series of asset returns
    Computes and returns a Dataframe that contains:
    the wealth index
    the prevoius peak
    percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns= (wealth_index- previous_peaks)/ previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top&Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
            header= 0, index_col= 0, na_values= -99.99)
    rets= me_m[['Lo 10','Hi 10']]
    rets.columns= ['Small Cap', 'Large Cap']
    rets= rets/100
    rets.index= pd.to_datetime(rets.index, format= '%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHED Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
            header= 0, index_col= 0, parse_dates= True)
    hfi= hfi/100
    hfi.index= hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header= 0, index_col= 0)/100 
    ind.index= pd.to_datetime(ind.index, format ='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header= 0, index_col= 0)
    ind.index= pd.to_datetime(ind.index, format ='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header= 0, index_col= 0)
    ind.index= pd.to_datetime(ind.index, format ='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    '''
    Computes the returns of a capweighted total market index for the 30 industry portfolio
    '''  
    # load the right returns 
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms*ind_size
    total_mktcap= ind_mktcap.sum(axis='columns')
    ind_capweight = ind_mktcap.divide(total_mktcap, axis= 'rows')#Compute that for every rows
    total_market_return = (ind_capweight * ind_return).sum(axis= 'columns')
    return total_market_return

def semideviation(r):
    """
    Returns the semi-deviation i.e. nagative semideviation of r
    r must be a Series or Dataframe
    """
    is_nagative = r<0
    return r[is_nagative].std(ddof=0)

def semideviation2(r):
    """
    Returns the semi-deviation i.e. negative semideviation of r
    r must be a Series or a DataFrame
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Compute the skewness of the given Series of DataFrame
    Returs a float or Series
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp= (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Compute the kurtosis of the given Series of DataFrame
    Returs a float or Series
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp= (demeaned_r**4).mean()
    return exp/sigma_r**4

import scipy.stats
def is_normal(r, level= 0.01):
    """
    Applies the J-B test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistics, p_value= scipy.stats.jarque_bera(r)
    return p_value > level

import numpy as np
def var_historic(r, level=5):
    """
    Return the VaR historic at a specified level
    i.e. returns the number such that 'level' percent of the returns fall below 
    that number, and the (100-level) percent are ablove
    """
    if isinstance(r, pd.DataFrame):  
        return r.aggregate(var_historic, level= level)
    elif isinstance(r, pd.Series): 
        return -np.percentile(r, level)
    else: 
        raise TypeError('Expected r to be a DataFrame or Series')

from scipy.stats import norm
def var_gaussian(r, level= 5, modified= False):
    """
    Return the Parametric Gaussian VaR of a Series or Dataframe
    If modified = True then return the modified VaR using the Cornish-Fisher modification
    """
    z= norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z= (z+
                (z**2 - 1)* s/6+
                (z**3 - z*3)*(k-3)/24 -
                (2*z**3 - z*5)*(s**2)/36
            
            )
    return -(r.mean()+ z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Compute the Conditional VaR of a Series or Dataframe
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level) #Find me all the returns < the historic VaR
        return -r[is_beyond].mean() #Then return the conditional mean 
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level= level)
    else:
        raise TypeError('Expected r to be a Dataframe or Series')
    

def annualized_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth **(periods_per_year/n_periods) -1

def annualized_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of the returns
    """
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r -rf_per_period
    ann_ex_ret = annualized_rets(excess_ret, periods_per_year)
    ann_vol = annualized_vol(r,periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Returnss
    """
    return weights.T @returns

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @covmat @weights)**0.5

def plot_ef2(n_points, er, cov, style= '.-'):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] !=2:
        raise ValueError('plot_ef2 can only plot 2-asset frontiers')
    weights= [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    efficient_frontier = pd.DataFrame({'Returns': rets, 'Volatility':vols})
    return efficient_frontier.plot.line(x='Volatility', y='Returns', style=style)

from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]                 #Number of assets
    init_guess = np.repeat(1/n, n)  #Initial guess
    bounds = ((0, 1), ) *n          #A sequence of bounds for every weights
    return_is_target = {
        'type': 'eq',  #equal
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1= {
        'type':'eq', #equal
        'fun': lambda weights: np.sum(weights) - 1
    }
    #Run the optimizer, which is to find a solution that satisfied the bound, constraints and run the objective function(minimize the vol)
    results = minimize(portfolio_vol, init_guess, 
                      args=(cov,), method="SLSQP", #This method = Quadratic Optimizer
                      options={'disp': False},
                      constraints = (return_is_target, weights_sum_to_1),
                      bounds = bounds 
                      )
    return results.x #pass back a variable collects in that structure


def optimal_weights(n_points, er, cov):
    """
    -> list of weights to run the optimizer on to minimize the vol
    """
    target_rets= np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rets]
    return weights

def msr(riskfree_rate, er, cov):
    """
    Returns the wieghts of the portfolio that gives you the max sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]                 #Number of assets
    init_guess = np.repeat(1/n, n)  #Initial guess
    bounds = ((0, 1), ) *n          #A sequence of bounds for every weights
    weights_sum_to_1= {
        'type':'eq', #equal
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the share ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol= portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol  

    results = minimize(neg_sharpe_ratio, init_guess,  #Minimizing the Negative Sharpe Ratio is exactly the same as maximizing the Sharpe Ratio
                      args=(riskfree_rate, er, cov,), method="SLSQP", #This method = Quadratic Optimizer
                      options={'disp': False},
                      constraints = (weights_sum_to_1),
                      bounds = bounds 
                      )
    return results.x #pass back a variable collects in that structure

def gmv(cov): 
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given the covariance matrix
    """
    n= cov.shape[0] #How many assets
    return msr(0, np.repeat(1,n), cov) #The risk-free rate does'nt matter when compute the GMV, so we can give a random number



def plot_ef(n_points, er, cov, style= '.-', show_cml= False, riskfree_rate=0, show_ew= False, show_gmv= False):
    """
    Plots the N-asset efficient frontier
    """
    weights= optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    efficient_frontier = pd.DataFrame({'Returns': rets, 'Volatility':vols})
    ax = efficient_frontier.plot.line(x='Volatility', y='Returns', style=style)
    if show_cml: #Capital Market Line
        ax.set_xlim(left= 0)
        w_msr = msr(riskfree_rate, er, cov) # Get the weights of the MSR Portfolio
        r_msr = portfolio_return(w_msr,er) 
        vol_msr = portfolio_vol(w_msr, cov)
        #Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color= 'green', marker= 'o', linestyle= 'dashed', markersize=12, linewidth= 2)

    if show_ew: #Equally Weighted Portfolio
        n= er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew= portfolio_vol(w_ew, cov)
        #Display EW
        ax.plot([vol_ew],[r_ew], color='gold', marker= 'o', markersize= 10)
    
    if show_gmv: #Gloval Minimum Volatility
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        #Display EW
        ax.plot([vol_gmv],[r_gmv], color='Cyan', marker= 'o', markersize= 10)
    
    return ax

def run_cppi(risky_r, safe_r= None, m= 3, start= 1000, floor= 0.8, riskfree_rate=0.03, drawdown= None):
    """
    Run a backtest of the CPPI strategy, give a set of returns for the risky asset
    Returns a dictionary containing: Asset Value history, Risk Budget history, Risk Weight History
    """
    #Set up the CPPI params
    dates= risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak= start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 
    #Set up ome DFs for saving intermediate values
    account_history=  pd.DataFrame().reindex_like(risky_r)
    cushion_history=  pd.DataFrame().reindex_like(risky_r)
    risky_w_history=  pd.DataFrame().reindex_like(risky_r)# risky weights

    for step in range(n_steps):
        if drawdown is not None:
            peak= np.maximum(peak, account_value)
            floor_value= peak*(1-drawdown)
        cushion= (account_value-floor_value) /account_value
        risky_w= m* cushion
        risky_w= np.minimum(risky_w, 1)# dont go above 100%
        risky_w= np.maximum(risky_w, 0)# dont go below 0
        safe_w = 1-risky_w
        risky_alloc = account_value * risky_w
        safe_alloc= account_value * safe_w
        # Update the account value for this time step
        account_value= risky_alloc * (1+risky_r.iloc[step]) + safe_alloc * (1+ safe_r.iloc[step])
        # Storing values so we can look at the history
        cushion_history.iloc[step]= cushion
        risky_w_history.iloc[step]= risky_w
        account_history.iloc[step]= account_value 

    #If we just put our $ in the risky asset
    risky_wealth = start*(1+risky_r).cumprod()

    #Return backtest result
    backtest_result= { 
        'Wealth': account_history, 
        'Risky Wealth': risky_wealth,
        'Risk Budget': cushion_history,
        'Risk Allocation': risky_w_history,
        'm': m,
        'start': start,
        'floor': floor,
        'risky_r': risky_r,
        'safe_r': safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate= 0.03):
    """
    Return a DF that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r= r.agg(annualized_rets, periods_per_year= 12)
    ann_vol= r.agg(annualized_vol, periods_per_year= 12)
    ann_sr= r.agg(sharpe_ratio, riskfree_rate= riskfree_rate, periods_per_year= 12)
    dd= r.agg(lambda r: drawdown(r).Drawdown.min())
    skew= r.agg(skewness)
    kurt= r.agg(kurtosis)
    cf_var5= r.agg(var_gaussian, modified= True)
    hist_var5= r.agg(cvar_historic)
    return pd.DataFrame({
        'Annualized Return': ann_r, 
        'Annualized Vol': ann_vol,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish-Fisher VaR(5%)': cf_var5,
        'Historic CVaR(5%)': hist_var5,
        'Sharpe Ratio': ann_sr,
        'Max Drawdown': dd
    })

#Geometric Brownian Motion
def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val
