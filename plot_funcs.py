import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def pretty_p_val(pval):
    if round(pval,3)<0.01:
        p_print="p< 0.01"
    else:
        val="%f" %pval
        p_print="p= " +val[0:4]
        print(p_print)
    return p_print

def correlation_plot (col_x, col_y, corr_dat, label, color):
    '''
    create plot of correlation; input is quoted name of x-column "column", y-column, and data (no quotes).
    label = whatever you want to label the plot
    '''
    #Calculate OLS linear fit
    lm1=smf.ols(col_y + '~' + col_x, data=corr_dat).fit() #fit
    #Make pretty p-value
    p_print=pretty_p_val(lm1.pvalues[1])
    #Store equation text for plot
    eq_text=col_y + "="+str(lm1.params[1])[0:4] + "x " + col_x + "+" + str(lm1.params[0])[0:6]+ "; R^2= " + str(lm1.rsquared_adj)[0:4] + "; " + p_print

    #Calculate Kendall Tau
    clean_dat=corr_dat.copy()
    clean_dat=corr_dat.dropna(axis=0, how='any')
    kendall_tau, kt_pval=scipy.stats.stats.kendalltau(clean_dat[col_y], clean_dat[col_x], nan_policy="omit")
    kt_pval=pretty_p_val(kt_pval)

    #plot and labels
    sns.jointplot(x=col_x, y=col_y, data=corr_dat, kind='reg', size=10, color=color, stat_func=kendall_tau)
    plt.ylabel(col_y+ ' (mm w.e.)')
    plt.xlabel(col_x + ' (mm w. e.)')
    plt.title(label)
    plt.tight_layout(pad=2)
    plt.figtext(0.93, 0.01, eq_text, horizontalalignment='right')
    plt.figtext(0.01, 0.01, "K. Tau = " + str(kendall_tau)[0:4] + "; " + kt_pval, horizontalalignment='right')

def OLS_plot(col_x, col_y, dat, hue=None, robust=False, title=None, color='blue', aspect=3):
    '''
    create correlation plot between two columns in a dataframe; add r2 and kendal tau stats to plot
    hue: name of column used to color the data points
    '''
    #Calculate correlation stats
   
    #OLS regression
    if robust==False:
        res=sm.OLS(dat[col_y], sm.add_constant(dat[col_x]), missing='drop').fit()
        pval=res.pvalues[col_x]
        r2=res.rsquared_adj
        slope=res.params[col_x]
    if robust:
        res=sm.RLM(dat[col_y], sm.add_constant(dat[col_x]), missing='drop').fit()
        pval=res.pvalues[col_x]
        r2=sm.OLS(dat[col_y], dat[col_x], missing='drop').fit().rsquared_adj #same r2 as for non-robust
        slope=res.params[col_x]

    #Kendal-Tau (non-parametric)
    kt_dat=dat.dropna(subset=[col_x, col_y])
    kendall_tau, kt_pval=scipy.stats.stats.kendalltau(kt_dat[col_y], kt_dat[col_x], nan_policy="omit")
    kt_pval=pretty_p_val(kt_pval)
    
    
    #Build plot
    sns.lmplot(y=col_y, x=col_x, data=dat, hue=hue, robust=robust,
               line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, scatter_kws={'color': color, 'alpha':0.6}, aspect=aspect)
    #plt.xticks(rotation=-90)
    summary_text="$r^2$=" +str(r2)[0:4]+ "; " +pretty_p_val(pval) + ". Slope= " + str(slope.round(4))
    plt.tight_layout(pad=2)
    plt.figtext(0.93, 0.01, summary_text, horizontalalignment='right')
    plt.figtext(0.02, 0.01, r"K. $\tau$ = " + str(kendall_tau)[0:5] + "; " + kt_pval, horizontalalignment='left')
    ax = plt.gca()
    ax.set_title(title)
    
    
    
    
    
    
    
    
    
    
    