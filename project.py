import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date as d
from datetime import time as t
from datetime import datetime as dt
import pydeck as pdk
import io
import base64

from project_utils import masscalc, windcalc, simcalc, el_pris, money, CAPEX, cost_methanol




####
## GENERAL SETTINGS
####

st.set_page_config(layout='wide')

hide_streamlit_style = """
            <style>
            /*#MainMenu {visibility: hidden;}*/
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Simulation error detector (for unimplemented functions)
sim_error = False

####
## SIDEBAR
####

st.sidebar.markdown(f"""
**Review the following columns and set the desired options.**\n
*** N.B. Do not close the containers after inputting data - this would reset the data. ***
""")

with st.sidebar.expander("Location Options"):
    input_lat = st.number_input('Wind Farm Latitude', min_value = 55.4272, max_value = 55.4272, value = 55.4272, format = "%.4f")
    input_lon = st.number_input('Wind Farm Longitude', min_value = 8.2119, max_value = 8.2119, value = 8.2119, format = "%.4f")

with st.sidebar.expander("System Options"):
    
    # Electrolyzer (Select Box)
    select_electrolyzer = st.selectbox('Electrolyzer Type', 
                        ['Alkaline (AEC)', 'Proton Exchange Membrane (PEMEC)', 'Solide Oxide (SOEC)'], index = 0, help = 'Which type of Electrolyzer one wants to use.')
    electrolyzer_warning = st.empty()
    if select_electrolyzer == 'Alkaline (AEC)':
        pass
    elif select_electrolyzer == 'Proton Exchange Membrane (PEMEC)':
        pass
    elif select_electrolyzer == 'Solide Oxide (SOEC)':
        pass
    else:
        electrolyzer_warning.error('This Electrolyzer has not been implemented. Please try again.')
        sim_error = True
        


    # Methanol Production Input (Number Input)
    input_prod_meth = st.number_input('Methanol Production (MW)', min_value = 10.0, max_value = 1000.0, 
                                            step =  0.001,
                                            format = "%.3f",
                                            help = 'The Methanol production capacity in MW.')

    mass_calculations = masscalc(input_prod_meth, select_electrolyzer)
    m_meth = mass_calculations[0] # kg
    tot_AC_pwr_req = mass_calculations[5]/1000 # MW
    tot_DC_pwr_req = mass_calculations[6] # kW



    # Automatic Calculation (On/Off)
    auto_onoff_help = f"""
    On: The Wind Farm capacity needed to produce exactly the given amount of methanol.\n\n
    Off: Manually define the Wind Farm capacity (the minimum value is the capacity required for the given amount of methanol)
    """
    select_auto_onoff = st.select_slider('Automatic Scaling:', options = ['Off', 'On'], value = 'On', help = auto_onoff_help)

    # Wind Capacity (Number Input - LOCKED if auto calculation on, else variable)
    input_wind_cap = st.number_input('Wind Farm Capacity (MW)', min_value = tot_AC_pwr_req, 
                                            max_value = tot_AC_pwr_req * 100 if select_auto_onoff == 'Off' else tot_AC_pwr_req,
                                            step =  0.001,
                                            format = "%.3f",
                                            help = 'The Wind Farm capacity - either automatically calculated or manually defined')

    # Wind Farm Type (Fixed/Variable)
    turbines_list = ['2019 NREL 12MW', 'DTU 10MW 178 RWT', 'LEANWIND 8MW 164 RWT', '2016 NREL 6MW', 'NREL 5MW 126 RWT']
    farm_type_help = f"""
    Which type of Wind Turbines one wants to use.\n
    Fixed: Choose one type of Turbine.\n
    Variable: A combination of Turbines is selected to match the capacity requirements.
    """
    select_farm_type = st.select_slider('Wind Farm Type:', options = ['Fixed', 'Variable'], value = 'Fixed', help = farm_type_help)
    farm_type_warning = st.empty()

    if select_farm_type == 'Fixed':
        # Fixed Turbine (Select Box)
        select_fixed_turbine = st.selectbox('Turbine Type', 
                                turbines_list,
                                index = 1)
        fixed_turbine_warning = st.empty()
        if select_fixed_turbine == 'DTU 10MW 178 RWT':
            pass
        else:
            fixed_turbine_warning.error('Turbine not implemented yet. Please select another turbine.')
            sim_error = True
    else:
        farm_type_warning.error('Variable farm type not implemented. Use fixed type instead.')
        sim_error = True

    distance_electrolyzer_meth = st.number_input('Electrolyzer - Methanol Plant Distance (m)', min_value = 1, max_value = 5000, value = 100, step = 1)
    distance_electrolyzer_powerplant = st.number_input('Electrolyzer - Power Plant Distance (m)', min_value = 1, max_value = 100000, value = 2000, step = 1)

    input_interest_rate = st.number_input('Interest Rate (%)', min_value = 1.0, max_value = 10.0, value = 3.5, step = 0.1, format = "%.1f")
    input_interest_rate /= 1e2 # % to decimal

with st.sidebar.expander("Sector Coupling"):

    # Frequency Regulation (On/Off)
    select_frequency_regulation = st.select_slider('Frequency Regulation', options = ['Off', 'On'], value = 'Off')
    frequency_regulation_warning = st.empty()
    if select_frequency_regulation == 'On':
        frequency_regulation_warning.error('Frequency Regulation not implemented yet. Please try again.')
        sim_error = True

    # District Heating (On/Off)
    select_district_heating = st.select_slider('District Heating:', options = ['Off', 'On'], value = 'Off')
    district_heating_warning = st.empty()
    if select_district_heating == 'On':
        district_heating_warning.error('District Heating not implemented yet. Please try again.')
        sim_error = True

    # Oxy-Fuel Combustion (On/Off)
    select_oxy = st.select_slider('Oxy-Fuel Combustion:', options = ['Off', 'On'], value = 'On')

    # Grid Connection (On/Off)
    select_grid = st.select_slider('Grid Connection:', options = ['Off', 'On'], value = 'On')

with st.sidebar.expander("Simulation Options"):

    select_simulation_range = st.select_slider('Simulation Range', options = ['Custom', 'Yearly'], value = 'Yearly', help = 'Select Range of Simulation - custom or year.')
    if select_simulation_range == 'Yearly':
        pass
    else:
        custom_date_start = st.date_input('Start Date', value = d(2019,1,1), min_value = d(2019, 1,1), max_value = d(2019, 12, 31))
        custom_time_start = st.time_input('Start Time', value = t(0,0,0))
        start_datetime_warning = st.empty()
        if custom_time_start.minute > 0 or custom_time_start.second > 0:
            start_datetime_warning.error('Start time is only available in resolution of hours.')
            sim_error = True
        start_datetime = dt.combine(custom_date_start, custom_time_start)

        custom_date_end = st.date_input('End Date', value = d(2019,12,31), min_value = d(2019, 1,1), max_value = d(2019, 12, 31))
        custom_time_end = st.time_input('End Time', value = t(23,0,0))
        end_datetime_warning = st.empty()
        if custom_time_end.minute > 0 or custom_time_end.second > 0:
            end_datetime_warning.error('End time is only available in resolution of hours.')
            sim_error = True
        end_datetime = dt.combine(custom_date_end, custom_time_end)



# Run Simulation (Button)
btn_run = st.sidebar.button('Run Simulation')

# Download Button
download_placeholder = st.sidebar.empty()



####
## MAIN
####

st.write("# Techno-Economic Analysis of Wind-Power to Methanol")
st.write("### Group 5 - 41418 Green Fuels and Power-To-X")

# Status Placeholder
status = st.empty()

# Metadata Section
section_metadata = st.expander('System Info', expanded = True)
metadata_placeholder = section_metadata.empty()
metadata_placeholder.info('Input data and run simulation first.')

meta_col_map, meta_col1, meta_col2 = section_metadata.columns(3)

# Yearly Section
section_yearly = st.expander('Yearly Production', expanded = True)
yearly_placeholder = section_yearly.empty()
yearly_placeholder.info('Input data and run simulation first.')

yearly_col1, yearly_col2, yearly_col3 = section_yearly.columns(3)

# Custom Section
section_custom = st.expander('Custom Interval Simulation', expanded = True)
if select_simulation_range == 'Custom':
    custom_placeholder = section_custom.empty()
    custom_placeholder.info('Input data and run simulation first.')

    custom_col1, custom_col2, custom_col3 = section_custom.columns(3)

# Expenditure Section
section_expenditure = st.expander('System Expenditure', expanded = False)
expenditure_placeholder = section_expenditure.empty()
expenditure_placeholder.info('Input data and run simulation first.')

if select_simulation_range == 'Custom':
    exp_metric_col1, exp_metric_col2, exp_metric_col3 = section_expenditure.columns(3)
else:
    exp_metric_col1, exp_metric_col2 = section_expenditure.columns(2)

exp_chart_col1, exp_chart_col2, exp_chart_col3 = section_expenditure.columns(3)

# Sources Section
section_sources = st.expander('Sources', expanded = False)
sources_placeholder = section_sources.empty()
sources_placeholder.info('Input data and run simulation first.')

if btn_run:
    if sim_error: # If any unimplemented functions
        status.error('You are trying to use unimplemented functions. Please try again.')
    else: # Main Simulation
        status.info('Running Simulation')

        year_pwr_prod, ts, actual_wind_capacity, n_turbines =  windcalc(input_wind_cap)
        year_pwr_prod = pd.DataFrame({'ts': ts, 'pwr_prod': year_pwr_prod})

        sim_df = simcalc(year_pwr_prod.pwr_prod, input_prod_meth, ts, select_electrolyzer)
        #grid_sales = el_pris(sim_df)
        money_df = money(sim_df, tot_DC_pwr_req, distance_electrolyzer_meth, distance_electrolyzer_powerplant, input_interest_rate, select_electrolyzer)
        capex_df = CAPEX(m_meth, actual_wind_capacity, tot_DC_pwr_req, distance_electrolyzer_powerplant, distance_electrolyzer_meth, sim_df, select_electrolyzer)



        # Methanol Cost
        meth_cost = cost_methanol(capex_df, money_df, input_interest_rate, sim_df, select_oxy = 'Off', select_grid = select_grid) # DKK/kg
        meth_cost_oxy = cost_methanol(capex_df, money_df, input_interest_rate, sim_df, select_oxy = 'On', select_grid = select_grid) # DKK/kg

        ####
        ## MODEL INFO
        ###

        model_df = pd.DataFrame({'Entity':
                                ['Methanol Plant', 'Electrolyzer', 'Theoretical Power Plant',
                                'Wind Farm - Requested', 'Wind Farm - Actual'],
                            'Capacity (MW)':
                                [input_prod_meth, mass_calculations[6]/1e3, mass_calculations[7]/1e3,
                                input_wind_cap, actual_wind_capacity/1e3]})
        wind_df = pd.DataFrame({'Turbine Type': turbines_list, 'Quantity': [int(0) if i != 'DTU 10MW 178 RWT' else int(n_turbines) for i in turbines_list]})

        metadata_placeholder.empty()
        yearly_placeholder.empty()
        if select_simulation_range == 'Custom':
            custom_placeholder.empty()
        expenditure_placeholder.empty()
        sources_placeholder.empty()
        
        map_df = pd.DataFrame({'lat': [input_lat], 'lon': [input_lon]})

        with st.container():
            with meta_col_map:
                st.markdown("Wind Farm Location")
                
                st.pydeck_chart(pdk.Deck(
                    map_style = 'mapbox://styles/mapbox/light-v9',
                    initial_view_state = pdk.ViewState(
                        latitude = input_lat,
                        longitude = input_lon,
                        zoom = 9,
                        pitch = 0,
                        use_container_width = True,

                    ),
                    layers = [
                        pdk.Layer(
                            'ScatterplotLayer',
                            data = map_df,
                            get_position = '[lon, lat]',
                            get_color = '[200, 30, 0, 160]',
                            get_radius = 500
                        )
                    ]
                ))


            with meta_col1:
                st.markdown("Plant Capacity")
                st.write(model_df)
            with meta_col2:
                st.markdown("Wind Farm Quantities")
                st.write(wind_df)

        ####
        ## YEARLY SIMULATION
        ###

        with st.container():

            with yearly_col1:

                st.metric('Wind Farm Production 2019', 
                f"{round(year_pwr_prod.pwr_prod.sum()/1e6,1)} GWh")

                st.metric('Potential Profits from Grid 2019' if select_grid == 'Off' else 'Profits from Grid 2019', 
                f"{round(money_df.sold_to_grid.sum()/1e6,1)} MDKK" if money_df.sold_to_grid.sum() > 1e6 else f"{round(money_df.sold_to_grid.sum()/1e3,1)} kDKK")


            with yearly_col2:
                st.metric('Hydrogen Production 2019',
                f"{round(sim_df['H2'].sum() / 1000,1)} t")

                st.metric('Cost of Methanol',
                f"{round(meth_cost,2)} DKK pr. kg")

            with yearly_col3:
                st.metric('Methanol Production 2019',
                f"{round(sim_df['Methanol'].sum() / 1000,1)} t")

                if select_oxy == 'On':
                    st.metric('Cost of Methanol w. Oxy-Fuel Combustion',
                    f"{round(meth_cost_oxy,2)} DKK pr. kg") 


        ####
        ## CUSTOM SIMULATION
        ###

        if select_simulation_range == 'Custom':
            with st.container():
                with custom_col1:

                    st.metric('Wind Farm Production in Period', 
                    f"""{round(
                            year_pwr_prod[
                                (year_pwr_prod.ts >= start_datetime) & (year_pwr_prod.ts <= end_datetime)
                                ]['pwr_prod'].sum()/1e6,1)} GWh""")

                    st.metric('Potential Profits from Grid in Period' if select_grid == 'Off' else 'Profits from Grid in Period', 
                    f"""{round(
                        money_df[
                            (money_df.ts >= start_datetime) & (money_df.ts <= end_datetime)
                            ]['sold_to_grid'].sum()/1e6,1)} MDKK""" 
                            if money_df[
                                (money_df.ts >= start_datetime) & (money_df.ts <= end_datetime)
                                ]['sold_to_grid'].sum() > 1e6 
                            else f"""{round(
                        money_df[
                            (money_df.ts >= start_datetime) & (money_df.ts <= end_datetime)
                            ]['sold_to_grid'].sum()/1e3,1)} kDKK"""
                        )
                


                with custom_col2:
                    st.metric('Hydrogen Production in Period',
                    f"""{round(
                            sim_df[
                                (sim_df.ts >= start_datetime) & (sim_df.ts <= end_datetime)
                                ]['H2'].sum() / 1000,1)} t""")


                with custom_col3:
                    st.metric('Methanol Production in Period',
                    f"""{round(
                        sim_df[
                            (sim_df.ts >= start_datetime) & (sim_df.ts <= end_datetime)
                            ]['Methanol'].sum() / 1000,1)} t""")

            with st.container():
                section_custom.write('Hour-by-Hour Simulation')

                section_custom.write(sim_df[(sim_df.ts >= start_datetime) & (sim_df.ts <= end_datetime)])


        ####
        ## EXPENDITURES
        ###

        with st.container():

            # CAPEX (20 year) metric
            with exp_metric_col1:
                lifetime_capex = capex_df.sum().sum()
                st.metric(
                    "CAPEX (over 20 years)",
                    f"{round(lifetime_capex / 1e9,2)} GDKK" if lifetime_capex > 1e9
                    else f"{round(lifetime_capex / 1e6,2)} MDKK"
                )

            # OPEX 2019 metric
            with exp_metric_col2:
                opex_df = money_df[['H2O_price','trans_H2','trans_O2','trans_CO2']]
                opex_2019 = opex_df.sum().sum()

                st.metric(
                    "OPEX 2019",
                    f"{round(opex_2019 / 1e9,2)} GDKK" if opex_2019 > 1e9
                    else f"{round(opex_2019 / 1e6,2)} MDKK"
                )

            if select_simulation_range == 'Custom':
                # OPEX Period metric
                with exp_metric_col3:
                    opex_period_df = money_df[(money_df.ts >= start_datetime) & (money_df.ts <= end_datetime)][['H2O_price','trans_H2','trans_O2','trans_CO2']]
                    opex_period = opex_period_df.sum().sum()

                    st.metric(
                        "OPEX for Period",
                        f"{round(opex_period / 1e9,2)} GDKK" if opex_period > 1e9
                        else (f"{round(opex_period / 1e6,2)} MDKK" if opex_period > 1e6 
                        else f"{round(opex_period / 1e3,2)} kDKK")
                    )

        with st.container():

            
            with exp_chart_col1:
                # CAPEX distribution
                st.write('CAPEX distribution')

                labels = np.array(['Electrolyzer', 'Wind Farm', 'Methanol Synthesis Plant', '$H_2$ Transport', '$O_2$ Transport', '$CO_2$ Transport'])
                sizes = np.array((capex_df.sum() / capex_df.sum().sum())*100)

                main = []
                others = []
                for i in range(len(sizes)):
                    if sizes[i] < 3.0:
                        others.append(i)
                    else:
                        main.append(i)
                plot_labels = np.append(np.array(labels)[main], 'Others')
                plot_sizes = np.append(np.array(sizes)[main], np.array(sizes[others]).sum())
                
                fig1, ax1 = plt.subplots(figsize = (5,5))
                ax1.pie(plot_sizes, labels = plot_labels, autopct = '%1.1f%%', shadow = True, startangle = 0)
                ax1.axis('equal')

                st.pyplot(fig1, transparent = True)
                

            
            with exp_chart_col2:
                # OPEX 2019 distribution
                st.write('OPEX 2019 distribution')

                labels = np.array(['H2O', 'H2 Transport', 'O2 Transport', 'CO2 Transport'])
                sizes = np.array((opex_df.sum() / opex_df.sum().sum())*100) 

                main = []
                others = []
                for i in range(len(sizes)):
                    if sizes[i] < 3.0:
                        others.append(i)
                    else:
                        main.append(i)
                plot_labels = np.append(np.array(labels)[main], 'Others')
                plot_sizes = np.append(np.array(sizes)[main], np.array(sizes[others]).sum())
                
                fig1, ax1 = plt.subplots(figsize = (5,5))
                ax1.pie(plot_sizes, labels = plot_labels, autopct = '%1.1f%%', shadow = True, startangle = 0)
                ax1.axis('equal')

                st.pyplot(fig1, transparent = True)
                
            
            with exp_chart_col3:
                # Profits from Sector Coupling 2019 Distribution
                st.write('Sector Coupling Profits 2019 distribution')


                

                if select_grid == 'Off' and select_oxy == 'Off':
                    st.write('No Data')
                else:

                    sold_list = []
                    labels = []
                    if select_oxy == 'On':
                        sold_list.append('sold_O2')
                        labels.append('O2 Sales')

                    if select_grid == 'On':
                        sold_list.append('sold_to_grid')
                        labels.append('Grid Sales')

                    sold_df = money_df[sold_list]

                    sizes = np.array((sold_df.sum() / sold_df.sum().sum())*100)

                    main = []
                    others = []
                    for i in range(len(sizes)):
                        if sizes[i] < 3.0:
                            others.append(i)
                        else:
                            main.append(i)

                    if len(main) == 0:
                        pass
                    elif len(main) == 1:
                        plot_labels = np.array([labels[main[0]]])
                        plot_sizes = np.array([sizes[main[0]]])
                    else:
                        if len(others) > 0:
                            plot_labels = np.append(np.array(labels)[main], 'Others')
                            plot_sizes = np.append(np.array(sizes)[main], np.array(sizes[others]).sum())
                        else:
                            plot_labels = np.array(labels)[main]
                            plot_sizes = np.array(sizes)[main]


                    fig1, ax1 = plt.subplots(figsize = (5,5))
                    ax1.pie(plot_sizes, labels = plot_labels, autopct = '%1.1f%%', shadow = True, startangle = 0)
                    ax1.axis('equal')

                    st.pyplot(fig1, transparent = True)
                

        capex_sources = pd.DataFrame(
            {
                'Item': ['Alkaline Electrolysis Cell', 'Wind Turbine', 'Methanol Synthesis', 
                'Transportation of H2', 'Transportation of CO2'],
                'Cost': ['8625', '22500', '967.25', 'Variable', '29.25'],
                'Unit': ['DKK/kW', 'DKK/kW', 'DKK/kg CH3OH', 'EUR/MW/m', 'DKK/km/[t CO2/h]'],
                'Link': ['https://www.sciencedirect.com/science/article/pii/S0306261920301847?fbclid=IwAR2pv3To7ZoD9Xr3ne_kK3q2QTN0CCzPHf0viq4CPM31gk1tqKsWaL2gS5A#b0395',
                         'https://drive.google.com/file/d/1kxPXXhv-lPEjeB0gClNlMlsfaCGcyMqw/view?usp=sharing',
                        'https://www.researchgate.net/publication/330501119_Technoeconomic_Assessment_of_Methanol_Synthesis_via_CO2_Hydrogenation',
                        'Energistyrelsen Teknologikatalog - Transport af Energi',
                        'Energistyrelsen Teknologikatalog - Transport af Energi']
            }
        )

        opex_sources = pd.DataFrame(
            {
                'Item': ['Water', 'Tranportation of CO2'],
                'Cost': ['0.5232', '150'],
                'Unit': ['DKK/kg H2', 'DKK/[t CO2/h]/yr/km'],
                'Link': ['https://theicct.org/sites/default/files/publications/final_icct2020_assessment_of%20_hydrogen_production_costs%20v2.pdf',
                         'Energistyrelsen Teknologikatalog - Transport af Energi ']
            }
        )

        eta_sources = pd.DataFrame(
            {
                'Item': ['Hydrogen', 'Wind Turbine Power Curves'],
                'Value': ['82', 'Variable'],
                'Unit': ['%', '%'],
                'Link': ['https://drive.google.com/file/d/1kxPXXhv-lPEjeB0gClNlMlsfaCGcyMqw/view?usp=sharing',
                         'https://nrel.github.io/turbine-models/Offshore.html?fbclid=IwAR0XY8kYWna7OE2bsqf3aVpF84MDXgvuvbnlwwb_-jYjanWQTHkUs5wMx_I'] 
            }
        )

        other_sources = pd.DataFrame({
            'Item': ['Wind Speed', 'Electricity Spot Price'],
            'Size': ['Variable', 'Variable'],
            'Unit': ['m/s', 'DKK/kWh/h'],
            'Link': ['https://www.renewables.ninja/',
                     'https://www.energidataservice.dk/']
        })

        section_sources.write('CAPEX Sources')
        section_sources.write(capex_sources)

        section_sources.write('OPEX Sources')
        section_sources.write(opex_sources)

        section_sources.write('Efficiencies Sources')
        section_sources.write(eta_sources)

        section_sources.write('Other Sources')
        section_sources.write(other_sources)




        with io.BytesIO() as output:
            with pd.ExcelWriter(output) as writer:
                model_df.to_excel(writer, sheet_name = 'Plant Information', index = False)
                wind_df.to_excel(writer, sheet_name = 'Wind Farm Quantities', index = False)
                sim_df.to_excel(writer, sheet_name = 'Yearly Simulation', index = False)
                
                if select_simulation_range == 'Custom':
                    custom_sim_df = sim_df[(sim_df.ts >= start_datetime) & (sim_df.ts <= end_datetime)]
                    custom_sim_df.to_excel(writer, sheet_name = 'Custom Simulation', index = False)

                capex_df.to_excel(writer, sheet_name = 'CAPEX', index = False)
                money_df.to_excel(writer, sheet_name = 'OPEX', index = False)     

            data = output.getvalue()

        b64 = base64.b64encode(data).decode("utf-8")
        href = f'<a download="methanol_plant_report_{dt.now()}.xlsx" href="data:file/excel;base64,{b64}">Download Complete Report as Excel File</a>'

        download_placeholder.markdown(href, unsafe_allow_html = True)

        status.success('Simulation Complete')
