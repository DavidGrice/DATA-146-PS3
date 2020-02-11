
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import math


# In[2]:


bostonData = pd.read_csv("Boston2014.csv")


# In[3]:


bostonData


# In[4]:


bostonData.describe()


# In[5]:


bostonTime = bostonData['official']
bostonTime.describe()


# In[6]:


sns.distplot(bostonTime)


# In[7]:


def sampling_dist_mean(data, size, num_reps):
    return [data.sample(size).values.mean() for i in range(num_reps)]
def sampling_dist_var(data, size, num_reps):
    return [data.sample(size).values.var(ddof=1) for i in range(num_reps)]


# In[8]:


bostonTimeMean = sampling_dist_mean(bostonTime, 5, 10000)
bostonTimeMean_series = pd.Series(bostonTimeMean)
bostonTimeMean_series.describe()


# In[9]:


sns.distplot(bostonTimeMean)


# In[10]:


sample_5 = bostonData['official'].sample(5)
#bostonTimeMean = sampling_dist_mean(bostonTime, 5, 1)
sample_5_standard_error = np.std(sample_5)/np.sqrt(5)
sample_5_standard_error
#bostonTimeMean_series = pd.Series(bostonTimeMean)
#bostonTimeMean_series.describe()


# In[11]:


bostonData['official'].std(ddof = 0)/5**0.5


# In[30]:


loveBoston = sampling_dist_var(bostonTime, 10000, 5)


# In[31]:


bostonTimeSD = np.sqrt(loveBoston)


# In[32]:


bostonTimeSD


# In[33]:


bostonTimeSDOne = 1/np.sqrt(5)


# In[34]:


bostonTimeSDOne


# In[35]:


bostonTimeT = bostonTimeMean/(bostonTimeSD/np.sqrt(10000))


# In[36]:


bostonTimeT


# In[37]:


sns.distplot(bostonTimeT)


# In[38]:


bostonTimeMean_graph = sampling_dist_mean(bostonTime, 20, 10000)
bostonTimeMean_series = pd.Series(bostonTimeMean_graph)
bostonTimeMean_series.describe()


# In[39]:


#Runners in 3-4 hours
bostonTimeRunners = sampling_dist_mean(bostonTime, 18, 10000)
bostonTimeMean_blank = pd.Series(bostonTimeRunners)
bostonTimeMean_blank.describe()


# In[40]:


running_mid = bostonTimeMean_blank.loc[bostonTimeMean_blank.between(180,240)]
running_mid.shape[0]/bostonTimeMean_blank.shape[0]


# In[41]:


boston_mid = bostonData.loc[bostonData['official'].between(180,240)]
boston_mid.shape[0]/bostonData.shape[0]


# In[42]:


boston_f = bostonData.loc[bostonData['gender'] == 'F']
boston_m = bostonData.loc[bostonData['gender'] == 'M']


# In[43]:


sample_f = sampling_dist_mean(boston_f['official'], 20, 10000)
sample_f_time = []

for i in sample_f:
    if i < 270:
        sample_f_time.append(i)


# In[44]:


len(sample_f_time)/len(sample_f)


# In[45]:


sample_m = sampling_dist_mean(boston_m['official'], 20, 10000)
sample_m_time = []

for i in sample_m:
    if i < 270:
        sample_m_time.append(i)

len(sample_m_time)/len(sample_m)


# In[46]:


male50 = bostonData.loc[(bostonData['gender'] == 'M') & (bostonData['age'] == 50)]


# In[47]:


male50


# In[48]:


male50_size = pd.Series(sampling_dist_var(male50['official'], 15, 10000))


# In[49]:


sns.distplot(male50_size)


# In[50]:


male50_size.describe()


# In[51]:


male50 #square the std.dev ofo fficial to asnwer f


# In[52]:


np.std(male50)


# In[53]:


np.mean(male50_size)


# In[54]:


(44.259430)**2 #estimate of variance for official


# In[55]:


male50_size_2 = pd.Series(sampling_dist_var(male50['official'], 15, 100000))
sns.distplot(male50_size_2)


# In[56]:


male50_size_3 = pd.Series(sampling_dist_var(male50['official'], 100, 100000))
sns.distplot(male50_size_3)
# More normally distributed


# In[57]:


presidentialData = pd.read_csv("Presidential_Age.txt", sep='\t', header = 0)


# In[58]:


presidentialData


# In[59]:


presidentialData.describe() #mean is 55.478965


# In[60]:


truePresidentialError = ss.sem(presidentialData['Real Age']) #true standard error
truePresidentialError


# In[133]:


cologne = [52.30479452, 55.34155251, 58.84703196, 68.06575342, 54.5630137, 70.602511]
meanOfMean = np.mean(cologne)


# In[134]:


np.std(cologne)


# In[135]:


boot_cologne = pd.Series((i for i in cologne))


# In[136]:


def bootstrap(data,num_reps,xbar=True):
    if xbar:
        return [np.mean(data.sample(len(data), replace=True).values) for i in range(num_reps)]
    return [np.median(data.sample(len(data), replace=True).values) for i in range(num_reps)]
    
def bootstrap_conf(estimates, alpha = 0.05):
    trim_pct = alpha/2
    new_arr = ss.trimboth(a = estimates, proportiontocut=trim_pct)
    return [new_arr[0], new_arr[-1]]

def bootstrap_conf_pivot(estimates, samp_est, alpha = 0.05): #also bootstrap_conf_delta
    diffs = np.array(estimates) - samp_est 
    new_arr = ss.trimboth(a = diffs, proportiontocut=alpha/2)
    return [samp_est-new_arr[-1], samp_est-new_arr[0]]


# In[137]:


cologne_mean = pd.Series(bootstrap(boot_cologne,10000,xbar=True))


# In[138]:


cologne_mean.describe()


# In[139]:


2.837361 / (np.sqrt(10000))


# In[140]:


ss.sem(cologne) #true standard error mean


# In[141]:


def t_conf(data, alpha = 0.05, xb=True):
    xbar = np.mean(data)
    if not xb:
      xbar = np.median(data)
    dof = len(data) - 1
    critical = 1-alpha/2
    moe = ss.t.ppf(critical,dof)*ss.sem(data)
    return [xbar - moe, xbar + moe]


# In[142]:


t_conf(cologne, alpha = 0.01, xb=True) # confidence interval


# In[143]:


presidentSample_six = presidentialData["Real Age"].sample(6)
presidentSample_six = presidentSample_six.astype(float)


# In[144]:


cologne_2 = pd.Series(bootstrap(presidentSample_six,10000,xbar=True))


# In[145]:


bootstrap_conf_pivot(cologne_2, 40, alpha = 0.08)


# In[146]:


def gen_intervals(data, reps = 25, sample_size = 20, alpha = 0.05, boot = False, boot_reps = None):
    intervals = []
    sample_means = []

    for i in range(reps):
        sample = np.random.choice(a= data, size = sample_size)
        sample_means.append(sample.mean())
        if(boot):
            estimates = bootstrap(pd.Series(sample), num_reps = boot_reps)
            confidence_interval = bootstrap_conf_pivot(estimates, sample.mean(), alpha)
        else:
            confidence_interval = t_conf(sample,alpha)
        intervals.append(confidence_interval)
    
    return (sample_means, intervals)
    

def plot_intervals(sample_stats, intervals, true_value):
    reps = len(intervals)
    plt.figure(figsize=(9,9))

    plt.errorbar(x=np.arange(0.1, reps, 1), 
             y=sample_stats, 
             yerr=[(top-bot)/2 for top,bot in intervals],
             fmt='o')

    plt.hlines(xmin=0, xmax=reps,
           y=true_value, 
           linewidth=2.0,
           color="red")


# In[147]:


testInt = gen_intervals(cologne, reps = 40, sample_size = 6, alpha = 0.08, boot = True, boot_reps = 1000)
testInt


# In[148]:


stat = testInt[0]
verval = testInt[1]
plot_intervals(stat, verval, 55.478965)


# In[149]:


#cologne_3 = pd.Series(bootstrap(boot_cologne,1000,xbar=True))
#standardErrorofCologne = np.std(cologne_3)/np.sqrt(6)
#standardErrorofCologne


# In[129]:


cologne_4 = pd.Series(bootstrap(boot_cologne,10000,xbar=True))
standardErrorofCologne = np.std(cologne_4)/np.sqrt(6)
standardErrorofCologne


# In[130]:


def boot_std_err(b_estimates, samp_estimate):   #b_estimates, list of estimates from bootstrap samples
    return (sum([(est - samp_estimate)**2 for est in b_estimates])/len(b_estimates))**.5


# In[150]:


bootStrapError = boot_std_err(cologne_4, meanOfMean)
bootStrapError


# In[132]:


trueStandardErrorCologne = (np.std(presidentialData['Real Age'])/np.sqrt(6))
trueStandardErrorCologne


# In[82]:


len(presidentialData['Real Age'])


# In[83]:


np.std(cologne)


# In[84]:


np.std(cologne_2)/np.sqrt(6)


# In[153]:


newcol = pd.Series([1567088.0, 1400000.0, 2993792.5699999998, 253286.0, 2058565.0, 4600000.0, 823204.0, 1728126.28, 2045181.0, 25498407.34])


# In[154]:


sns.distplot(newcol)


# In[155]:


newcol.describe() # mean is 4.29765x10^6


# In[88]:


sample_pivot = ss.sem(newcol)


# In[89]:


np.median(newcol) # median of sample


# In[158]:


newcolBoot = pd.Series(bootstrap(newcol,10000,xbar=False))
newcolBoot.describe()


# In[91]:


sns.distplot(newcolBoot)


# In[92]:


np.median(newcolBoot)


# In[93]:


np.median(newcol)


# In[94]:


np.std(newcolBoot)/np.sqrt(len(newcolBoot))


# In[95]:


std_err = 0


# In[96]:


for i in newcolBoot:
    mean = (i - np.median(newcol))**2
    std_err += mean


# In[97]:


np.sqrt(std_err/(len(newcol)))


# In[98]:


bootstrap_conf(newcolBoot, alpha = 0.10)


# In[99]:


bootstrap_conf_pivot(newcolBoot, np.median(newcol), alpha = 0.10) # confidence interval for data


# In[100]:


t_conf(newcol, alpha = 0.05, xb=True)


# In[101]:


endorse = pd.read_csv('endorsements-june-30.csv')


# In[102]:


showMeThaMoney = pd.Series(endorse['money_raised']).sample(10)


# In[103]:


showMeThaMoney.describe()


# In[104]:


sns.distplot(showMeThaMoney)


# In[105]:


np.median(showMeThaMoney)


# In[106]:


showMeThaBootstrap = pd.Series(bootstrap(showMeThaMoney,10000,xbar=False))
showMeThaBootstrap.describe()


# In[107]:


sns.distplot(showMeThaBootstrap)


# In[108]:


np.median(showMeThaBootstrap)


# In[109]:


pennyPincher = 0
for i in showMeThaBootstrap:
    meanOfMoney = (i - np.median(showMeThaMoney))**2
    pennyPincher += meanOfMoney


# In[110]:


np.sqrt(pennyPincher/(len(showMeThaMoney))) #standard error for show me tha money AKA endorsement of random sample size 10

