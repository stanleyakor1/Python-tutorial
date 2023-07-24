import sys
import urllib3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.options.mode.chained_assignment = None

def getData(SiteName, SiteID, StateAbb, StartDate, EndDate):
	url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/'
	url2 = f'{SiteID}:{StateAbb}:SNTL%7Cid=%22%22%7Cname/'
	url3 = f'{StartDate}-10-01,{EndDate}-09-30/'
	url4 = 'PREC::value,PRCP::value,PRCPSA::value,WTEQ::value,SNRR::value?fitToScreen=false'
	url = url1+url2+url3+url4
	
	print(f'Start retrieving data for {SiteName}, {SiteID}')
	
	http = urllib3.PoolManager()
	response = http.request('GET', url)
	data = response.data.decode('utf-8')
	i=0
	for line in data.split("\n"):
		if line.startswith("#"):
			i=i+1
	data = data.split("\n")[i:]
	
	df = pd.DataFrame.from_dict(data)
	df = df[0].str.split(',', expand=True)
	df.rename(columns={0:df[0][0], 
					   1:df[1][0], 
					   2:df[2][0],
					   3:df[3][0],
					   4:df[4][0],
					   5:df[5][0]}, inplace=True)
	df.drop(0, inplace=True)
	df.dropna(inplace=True)
	df.reset_index(inplace=True, drop=True)
	df["Date"] = pd.to_datetime(df["Date"])
	df[f'{SiteName} ({SiteID}) Precipitation Accumulation (in) Start of Day Values'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Precipitation Accumulation (in) Start of Day Values'])
	df[f'{SiteName} ({SiteID}) Precipitation Increment (in)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Precipitation Increment (in)'])
	df[f'{SiteName} ({SiteID}) Precipitation Increment - Snow-adj (in)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Precipitation Increment - Snow-adj (in)'])
	df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'])
	df[f'{SiteName} ({SiteID}) Snow Rain Ratio (unitless)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Snow Rain Ratio (unitless)'])
	df['SiteID'] = SiteID
	df['Water_Year'] = pd.to_datetime(df['Date']).map(lambda x: x.year+1 if x.month>9 else x.year)
	
	df['CALCULATED Precipitation Accumulation (in)'] = df.groupby(['Water_Year', 'SiteID'])[f'{SiteName} ({SiteID}) Precipitation Increment (in)'].cumsum() 
	df['CALCULATED Snow Adjusted Precipitation Accumulation (in)'] = df.groupby(['Water_Year', 'SiteID'])[f'{SiteName} ({SiteID}) Precipitation Increment - Snow-adj (in)'].cumsum() 

	df['OBSERVED Snow Rain Ratio (unitless)'] = np.zeros(df.shape[0])
	df['OBSERVED Snow Rain Ratio (unitless)'][:] = np.NaN
	for i in range(0, len(df.index)-1):
		if df[f'{SiteName} ({SiteID}) Precipitation Increment - Snow-adj (in)'][i]>0:
			if df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'][i+1]>=df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'][i]:
				df['OBSERVED Snow Rain Ratio (unitless)'][i] = (df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'][i+1] - df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'][i])/df[f'{SiteName} ({SiteID}) Precipitation Increment - Snow-adj (in)'][i]
			else:
				df['OBSERVED Snow Rain Ratio (unitless)'][i] = 0
		else:
			df['OBSERVED Snow Rain Ratio (unitless)'][i] = np.nan
	
	df.to_csv(f'./df_{SiteID}.csv', index=False)
	
if __name__ == "__main__":
	SiteName = sys.argv[1]
	SiteID = sys.argv[2]
	StateAbb = sys.argv[3]
	StartDate = sys.argv[4]
	EndDate = sys.argv[5]
	
	getData(SiteName, SiteID, StateAbb, StartDate, EndDate)