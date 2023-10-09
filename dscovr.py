import pandas
import xarray as xr

xr.set_options(display_max_rows = 1024, display_style = 'text', display_width = 1024)

filename = "ncdata/dsc_fc_summed_spectra_2021_v01.csv"
data = pandas.read_csv(filename, \
                       delimiter=',', \
                       parse_dates=[0], \
                       date_format="YYYY-MM-DD hh:mm:ss", \
                       na_values='0', \
                       header = None)

df = pandas.DataFrame({'t': data[0], 'B[x]': data[1], 'B[y]': data[2], 'B[z]': data[3]})

att = "ncdata/oe_att_dscovr.nc"
fim = "ncdata/oe_f1m_dscovr.nc"
f3s = "ncdata/oe_f3s_dscovr.nc"
fc0 = "ncdata/oe_fc0_dscovr.nc"
fc1 = "ncdata/oe_fc1_dscovr.nc"
m1m = "ncdata/oe_m1m_dscovr.nc"
m1s = "ncdata/oe_m1s_dscovr.nc"
mg0 = "ncdata/oe_mg0_dscovr.nc"
mg1 = "ncdata/oe_mg1_dscovr.nc"
pop = "ncdata/oe_pop_dscovr.nc"
rt0 = "ncdata/oe_rt0_dscovr.nc"
rt1 = "ncdata/oe_rt1_dscovr.nc"
vc0 = "ncdata/oe_vc0_dscovr.nc"
vc1 = "ncdata/oe_vc1_dscovr.nc"

ds1 = xr.open_dataset(fc1)
# ds1c = xr.open_dataset(fcc)
ds2 = xr.open_dataset(m1s)

df = ds1.to_dataframe()
df.to_csv('ds1.csv')
