import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

def masscalc(prod_meth, electrolyzer): # input in MW
    # Methanol
    energy_density_meth = 4.4 # kWh / L
    V_meth = (prod_meth*10**3) / energy_density_meth # L

    rho_meth = 0.792 # kg/L
    m_meth = rho_meth * V_meth # kg

    M_meth = 0.032 # kg/mol
    n_meth = m_meth / M_meth # mol

    # Hydrogen
    eta_H2_meth = 1.0

    n_H2 = n_meth * 3 # mol
    M_H2 = 0.002 # kg/mol
    m_H2 = n_H2 * M_H2 * (1/eta_H2_meth) # kg

    # CO2
    eta_CO2_meth = 1.0

    n_CO2 = n_meth # mol
    M_CO2 = 0.044 # kg/mol
    m_CO2 = n_CO2 * M_CO2 * (1/eta_CO2_meth) # kg

    # Water produced in methanol production
    n_H2O_meth = n_meth # mol
    M_H2O_meth = 0.02 # kg/mol
    m_H2O_meth = n_H2O_meth * M_H2O_meth # kg
    
    # Water for electrolysis
    n_H2O_el = n_H2 # mol
    M_H2O_el = 0.02 # kg/mol
    
    if electrolyzer == 'Solide Oxide (SOEC)':
        m_H2O_el = (1/0.8) * ((n_H2O_el * M_H2O_el) - m_H2O_meth) # kg
    else:
        m_H2O_el = (n_H2O_el * M_H2O_el) - m_H2O_meth # kg

    # Mass of oxygen produced at electrolysis
    n_O2 = n_H2 / 2 # mol
    M_O2 = 0.032 # kg / mol
    m_O2 = n_O2 * M_O2 # kg

    # Required DC electricity for electrolysis

    HHV_H2 = 33.33 # kWh/kg # Actually Lower Heating Value
    
    # Efficiency of electrolysis (varies depending on electrolyzer type)
    if electrolyzer == 'Alkaline (AEC)':
        eta_el_H2_HHV = 0.7
    elif electrolyzer == 'Proton Exchange Membrane (PEMEC)':
        eta_el_H2_HHV = 0.7
    elif electrolyzer == 'Solide Oxide (SOEC)':
        eta_el_H2_HHV = 0.9
    else:
        eta_el_H2_HHV = 0.5
    
    DC_pwr_req = HHV_H2 / eta_el_H2_HHV # kWh / kg H2

    tot_DC_pwr_req = DC_pwr_req * m_H2 # kWh

    # Power Plant size

    # # Electrolysis side
    rho_O2 = 1.314 # kg/m^3
    power_plant_req_O2 = 24146 # m^3/h
    m_O2_in_power_plant_unit = power_plant_req_O2 * rho_O2 # kg/h
    ref_power_plant_capacity = 83.1 * 10**3 # kW
    m_O2_in_power_plant = m_O2_in_power_plant_unit / ref_power_plant_capacity # kg / kWh

    O2_out_to_electrolysis_ratio = m_O2 / tot_DC_pwr_req # kg / kWh
    electrolysis_to_power_plant_ratio = m_O2_in_power_plant / O2_out_to_electrolysis_ratio


    # # Methanol side
    eta_CO2_out_power_plant = 0.95 
    power_plant_prod_CO2_unit = 14405 * eta_CO2_out_power_plant # m^3/h
    power_plant_prod_CO2 = power_plant_prod_CO2_unit / ref_power_plant_capacity # m^3/kWh

    rho_CO2 = 1.87 # kg/m^3
    m_CO2_out_power_plant = power_plant_prod_CO2 * rho_CO2 # kg / kWh

    meth_to_power_plant_ratio = m_CO2_out_power_plant / (m_CO2/prod_meth) 


    electrolysis_to_power_plant_size = tot_DC_pwr_req * (1/electrolysis_to_power_plant_ratio) # kWh
    meth_to_power_plant_size = prod_meth * (1/meth_to_power_plant_ratio) # kWh

    power_plant_size = max(electrolysis_to_power_plant_size, meth_to_power_plant_size) # kWh


    # AC/DC Converter
    eta_ACDC = 0.9

    tot_AC_pwr_req = tot_DC_pwr_req * (1/eta_ACDC)

    return m_meth,m_H2,m_CO2,m_H2O_el,m_O2,tot_AC_pwr_req,tot_DC_pwr_req, power_plant_size


def turbine_data(url):
    url = url
    df = pd.read_csv(url)
    df.rename(
            columns = {
                df.columns[0]: 'wind_speed',
                df.columns[1]: 'power',
                df.columns[2]: 'cp'
            },
            inplace = True
    )
    
    df.drop(columns = [df.columns[3], df.columns[4]], inplace = True)
    
    return df

def power(df, x):
    X = np.array(df.wind_speed).reshape(-1,1)
    y = np.array(df.power).reshape(-1,1)
    
    x = x[:, None]
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree = 8)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    
    output = lin2.predict(poly.fit_transform(x))
    
    output[np.where((x < 4) | (x > 25))[0]] = 0.0
    
    return output.flatten()

def windcalc(farm_size): # in MW

    df = turbine_data("https://raw.githubusercontent.com/NREL/turbine-models/master/Offshore/DTU_10MW_178_RWT_v1.csv")

    numberofturbines = np.ceil((farm_size*10**3)/df.power.max()) # Using 10 MW turbines
    
    actual_cap = numberofturbines*df.power.max()

    import requests
    import pandas as pd
    import json

    token = '2a418402d52939fa6ffee527cbf8344946800efc'
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()

    s.headers = {'Authorization': 'Token ' + token}

    url = api_base + 'data/wind'

    args = {
        'lat': 55.4272,
        'lon': 8.2119,
        'date_from': '2019-01-01',
        'date_to': '2019-12-31',
        'capacity': 1.0,
        'height': 100,
        'turbine': 'Vestas V80 2000',
        'raw': True,
        'format': 'json'
    }

    try:
        r = s.get(url, params=args)

        parsed_response = json.loads(r.text)
        data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
        print('wind data from api')
    except:
        data = pd.read_csv('wind_data.csv', sep = ',')
        print('wind data from csv')
    

    pwr = power(df, np.array(data.wind_speed))
    return pwr * numberofturbines,data.index,actual_cap, numberofturbines

def simcalc(pwr_prod, prod_meth, ts, electrolyzer):
    
    m_meth,m_H2,m_CO2,m_H2O_el,m_O2,tot_AC_pwr_req,tot_DC_pwr_req, power_plant_size = masscalc(prod_meth, electrolyzer)

    AC_DC = 0.9 
    DC_H2 = tot_DC_pwr_req/m_H2 #kWh/kg H2
    H2_meth = m_meth/m_H2 # kg meth/kg H2
    H2_O2 = m_O2/m_H2 #kWh/kg O2
    H2O_H2 = m_H2O_el/m_H2 # kg H2O/kg H2
    CO2_meth = m_CO2/m_meth # kg CO2/kg meth    
    
    sim_df = pd.DataFrame(pwr_prod)
    sim_df.columns = ['pwr_prod']

    #Export to grid
    sim_df.loc[sim_df['pwr_prod'] > tot_AC_pwr_req, 'Export to grid']=sim_df['pwr_prod']-tot_AC_pwr_req
    sim_df=sim_df.replace({np.NaN: 0})

    # DC pwr produced
    sim_df['DC_pwr'] = AC_DC * sim_df['pwr_prod']

    # H2 produced
    sim_df['H2'] = sim_df['DC_pwr']/ DC_H2

    # H2O used
    sim_df['H2O'] = H2O_H2 * sim_df['H2']

    # O2 produced
    sim_df['O2'] = H2_O2 * sim_df['H2']

    # methanol produced
    sim_df['Methanol'] = H2_meth * sim_df['H2']

    # CO2 used
    sim_df['CO2'] = CO2_meth * sim_df['Methanol']

    sim_df['ts'] = ts
    
    return sim_df

def transport(cap,distance, electrolyzer): # input total DC required
    cap*=0.82
    if cap <= 100:
        price = 3.9 * 7.5 # DKK/MW/m
    elif 100 < cap <= 250:
        price = 1.7 * 7.5
    elif 250 < cap <= 500:
        price = 1.1 * 7.5
    elif 500 < cap <= 1000:
        price =0.7 * 7.5
    elif 1000 < cap <= 4000:
        price = 0.4 * 7.5
    else:
        price = 0.2 * 7.5
    capex = price * distance * cap

    opex1 = 0.25 * 7.5 # DKK/km/year/MW

    HHV_H2 = 33.33 # kWh/kg # Actually Lower Heating Value, just didn't change the name

    if electrolyzer == 'Alkaline (AEC)':
        eta_el_H2_HHV = 0.7
    elif electrolyzer == 'Proton Exchange Membrane (PEMEC)':
        eta_el_H2_HHV = 0.7
    elif electrolyzer == 'Solide Oxide (SOEC)':
        eta_el_H2_HHV = 0.9
    else:
        eta_el_H2_HHV = 0.5

    DC_pwr_req = HHV_H2 / eta_el_H2_HHV # kWh / kg H2

    opex2 = (opex1/1000)/DC_pwr_req # DKK/km/year/kg
    opex_final = (opex2/1000) * distance # DKK/kg
    
    return opex_final,capex


def oxy(sim_df, r):
## Calculating potential oxygen income 
    pp_opex = (sim_df.O2.sum()*48/1000)

    pp_capex = (sim_df.O2.max()/1000)*900000*7.45

    # capex med rente
    pp_capex_rente = (pp_capex * (1+r)**20)/20

    pot_income = pp_capex_rente + pp_opex

    ## Oxygen salesprice
    O2_price = (pot_income * 0.66)/sim_df.O2.sum() # DKK/kg O2

    return O2_price

def money(sim_df, tot_DC_pwr_req, distance_H2, distance_O2, r, electrolyzer):
    
    # Electricity Price
    import requests
    import json

    url = "https://api.energidataservice.dk/datastore_search_sql?sql="

    query = """
    SELECT
        "HourDK",
        "PriceArea",
        "SpotPriceDKK"
    FROM "elspotprices"
    WHERE 
        "PriceArea" = 'DK1'
    AND
        "HourDK" >= '2019-01-01T00:00:00'
    AND
        "HourDK" <= '2020-01-01T00:00:00'
    ORDER BY
        "HourDK" ASC
    """
    
    try:
        response = requests.get(url+query)
        el_spot = pd.DataFrame(json.loads(response.text)['result']['records'])
    except:
        el_spot = pd.read_csv('el_spot_prices.csv')
        print('spot price data from csv')

    if len(el_spot) > 8760:
        el_spot.drop(8760, inplace = True)
    ## revenue if sold to grid
    rev_grid = np.array(sim_df['Export to grid']/1000)*(el_spot['SpotPriceDKK'])

    money_df = pd.DataFrame(rev_grid)
    money_df.columns = ['rev_grid']
    ## remove all negative values, sold_to_grid is what's actually sold to the grid 
    money_df['sold_to_grid'] = [0 if i < 0 else i for i in money_df.rev_grid]

    money_df['ts'] = sim_df.ts
    
    # Water
    H2O_price = 0.08 * 6.54 # convert from USD to DKK

    money_df['H2O_price'] = sim_df.H2 * H2O_price
    
    # Transport
    ## OPEX transport Hydrogen
    op_H2,_ = transport(tot_DC_pwr_req,distance_H2, electrolyzer)
    money_df['trans_H2'] = sim_df.H2*op_H2
    
    ## OPEX transport Oxygen
    op_O2,_ = transport(tot_DC_pwr_req,distance_O2, electrolyzer)

    money_df['trans_O2'] = sim_df.O2*op_O2
    
    O2_price = oxy(sim_df, r)
    money_df['sold_O2'] = sim_df.O2 * O2_price
    
    ## OPEX transport CO2
    distance_CO2 = distance_O2
    opex1 = 20 * 7.5 # DKK/[t co2/h]/year/km

    opex2 = opex1 * distance_CO2/1000 # DKK/[t CO2/h]

    CO2_temp = np.empty(len(money_df));CO2_temp.fill((sim_df.CO2.max()/1000 * opex2)/len(money_df))

    money_df['trans_CO2'] = CO2_temp # DKK
    
    
    return money_df

def CAPEX(m_meth,actual_cap,tot_DC_pwr_req,distance_O2,distance_H2,sim_df, electrolyzer):
    
    # Electrolyzer cell costs
    if electrolyzer == 'Alkaline (AEC)':
        Alk_EC_price = 1000 * 7.5 # DKK/kW
        Alk_EC_CAPEX = Alk_EC_price * tot_DC_pwr_req
    elif electrolyzer == 'Proton Exchange Membrane (PEMEC)':
        Alk_EC_price = 1600 * 7.5 # DKK/kW
        Alk_EC_CAPEX = Alk_EC_price * tot_DC_pwr_req
    elif electrolyzer == 'Solide Oxide (SOEC)':
        Alk_EC_price = 4000 * 7.5 # DKK/kW
        Alk_EC_CAPEX = Alk_EC_price * tot_DC_pwr_req
    else:
        Alk_EC_price = 2200 * 7.5 # DKK/kW
        Alk_EC_CAPEX = Alk_EC_price * tot_DC_pwr_req


    # Wind farm
    turbine_price = 3000 * 7.5 #DKK/kW
    wind_price = turbine_price * actual_cap

    # Methanol Synthesis

    meth_unit_price = (129.03 * 7.5)# DKK/kg 

    meth_price = meth_unit_price * m_meth

    # transport H2
    opex_trans,capex_trans_H2 = transport(tot_DC_pwr_req,distance_H2, electrolyzer)

    # transport O2
    opex_trans,capex_trans_O2 = transport(tot_DC_pwr_req/1000,distance_O2, electrolyzer)

    # transport CO2
    max_massflow_CO2 = sim_df.CO2.max()/1000
    capex_trans_CO2 = 3.9*max_massflow_CO2*(distance_O2/1000)*7.5

    # Construct dataframe

    capex_df = pd.DataFrame(['Alkaline EC',round(Alk_EC_CAPEX,2)])
    capex_df.columns = ['EC_plant']

    capex_df['wind_farm'] = round(wind_price,2)
    capex_df['meth_plant'] = round(meth_price,2)
    capex_df['trans_H2'] = round(capex_trans_H2,2)
    capex_df['trans_O2'] = round(capex_trans_O2,2)
    capex_df['trans_CO2'] = round(capex_trans_CO2,2)

    capex_df.drop(0,inplace = True)
    return capex_df


def cost_methanol(capex_df,money_df,r,sim_df, select_oxy, select_grid): # r is interest rate
    opex_df = money_df[['H2O_price','trans_H2','trans_O2','trans_CO2']]
    
    sold_list = []
    if select_oxy == 'On':
        sold_list.append('sold_O2')
    
    if select_grid == 'On':
        sold_list.append('sold_to_grid')

    sold_df = money_df[sold_list]

    capex = capex_df.sum().sum()
    opex = opex_df.sum().sum()
    sold = sold_df.sum().sum()

    price = (-sold+opex+((capex*(1+r)**20)/20))/sim_df.Methanol.sum()

    return price


def el_pris(sim_df):
    
    # Electricity Price
    import requests
    import json

    url = "https://api.energidataservice.dk/datastore_search_sql?sql="

    query = """
    SELECT
        "HourDK",
        "PriceArea",
        "SpotPriceDKK"
    FROM "elspotprices"
    WHERE 
        "PriceArea" = 'DK1'
    AND
        "HourDK" >= '2019-01-01T00:00:00'
    AND
        "HourDK" <= '2020-01-01T00:00:00'
    ORDER BY
        "HourDK" ASC
    """
    response = requests.get(url+query)
    el_spot = pd.DataFrame(json.loads(response.text)['result']['records'])

    if len(el_spot) > 8760:
        el_spot.drop(8760, inplace = True)
    ## revenue if sold to grid
    rev_grid = np.array(sim_df['Export to grid']/1000)*(el_spot['SpotPriceDKK'])

    money_df = pd.DataFrame(rev_grid)
    money_df.columns = ['rev_grid']
    ## remove all negative values, sold_to_grid is what's actually sold to the grid 
    money_df['sold_to_grid'] = [0 if i < 0 else i for i in money_df.rev_grid]

    money_df['ts'] = sim_df.ts
    





    return money_df
