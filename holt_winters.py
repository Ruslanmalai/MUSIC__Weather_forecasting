import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize              # for function minimization
#import matplotlib.pyplot as plt


class HoltWinters(object):
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, slen, n_preds, train_hours, scaling_factor=1.96):
        #self.series = series
        self.slen = slen
#        self.alpha = alpha
#        self.beta = beta
#        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.train_hours = train_hours
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])
            self.train_result = self.result[:-self.n_preds]
            

    def CVscore(self, params, series, loss_function = mean_squared_error):
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        
        errors = []
        values = series.values
        
        num_splits = 10  
        #num_splits = int(len(series)/self.slen)
        #tscv = TimeSeriesSplit(n_splits = num_splits, max_train_size = self.train_hours)
        tscv = TimeSeriesSplit(n_splits = num_splits)
        
        
        # iterating over folds, train model on each, forecast and calculate error
        for train, test in tscv.split(values):

        #    model = HoltWinters(series=values[train], slen=slen, 
        #                    alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
            self.series = values[train]
            self.n_preds = len(test)
            self.triple_exponential_smoothing()
        
            predictions = self.result[-len(test):]
            actual = values[test]
            error = loss_function(predictions, actual)
            errors.append(error)
        
        self.train_error = np.sqrt(np.mean(np.array(errors)))
        return np.mean(np.array(errors))
    
    def fit(self, series):
        # initializing model parameters alpha, beta and gamma
        self.train_series = series
        self.series = series
        x = [0, 0, 0]
        
        # Minimizing the loss function 
        opt = minimize(self.CVscore, x0=x, 
                       args=(self.series, mean_squared_error), 
                       method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                       )
        # Take optimal values...
        self.alpha, self.beta, self.gamma = opt.x
        
#    def predict(self):
#        self.series = self.train_series
#        self.triple_exponential_smoothing()
#        return self.result[-self.n_preds:]
        
    def predict(self, series, n_preds):
        self.series = series
        self.n_preds = n_preds
        self.triple_exponential_smoothing()
        return self.result[-self.n_preds:]
         
        
        
        
        
        
        
        
        
    
        
        
        
        