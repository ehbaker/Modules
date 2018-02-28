'''
  contains functions to work with date-time issues in Python
'''


def add_water_year_column(dat):
    #Look at the data by Water Year
    dat['wy']=np.nan #create empty water year column
    dat.wy=dat.index.year
    dat.wy[(dat.index.month==10) | (dat.index.month==11) | (dat.index.month==12)]=dat.wy+1
    return(dat)