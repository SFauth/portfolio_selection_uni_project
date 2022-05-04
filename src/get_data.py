#%% Get DSWS access
import DatastreamDSWS as DSWS

ds = DSWS.Datastream(username='stefan.fauth@student.uni-tuebingen.de', password="abc")

# %%
ds.get_data(tickers='VOD', fields='P', kind=0)
# %%
