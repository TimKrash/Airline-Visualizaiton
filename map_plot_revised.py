import pandas as pd
import os
import networkx as  nx
import numpy as np
import sqlite3
import calendar
import csv
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networka.structure as ns
import or_conditions_TSJ2 as oc

## Need to import this to establish DB connection
from sqlalchemy import *


def create_table():
    conn = sqlite3.connect('atn_performance.db')
    c = conn.cursor()
    
    ##Change column names to match the ones from csv
    c.execute('CREATE TABLE IF NOT EXISTS atn_performance(Origin_Airport_Code TEXT, Destination_Airport_Code TEXT)')

def data_entry():

    cwd = os.getcwd()
    script_dir = os.path.dirname(os.getcwd())
    rel_path = "data/processed/%s_%s_combined.csv" % ('2008', 'AA')
    raw_file = os.path.join(script_dir, rel_path)
    
    ##Remove contry from the import fields since you don't use it in the DB
    ##Change the name to not fields, that create an error
    import_cols = ['Origin_Airport_Code', 'Destination_Airport_Code']

    df = pd.read_csv(raw_file,usecols = import_cols)
    
    ##Use the built-in pandas function to improt the df directly into the database
    ##Be sure to establish connection to the DB beforehand
    conn = sqlite3.connect('airport_table.db')
    engine = create_engine('sqlite:///airport_table.db')
    
    pd.DataFrame.to_sql(self = df, name = 'airportTable', con=engine, if_exists='append', index=False,chunksize=100000)
    
    conn.commit()
    conn.close()
"""
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.getcwd())
    rel_path = "data/processed/%s.csv" % 'airport_data'
    raw_file = os.path.join(script_dir, rel_path)
    
    ##Remove contry from the import fields since you don't use it in the DB
    ##Change the name to not fields, that create an error
    import_cols = ['IATA', 'lat', 'long']

    df = pd.read_csv(raw_file,usecols = import_cols)
    
    ##Use the built-in pandas function to improt the df directly into the database
    ##Be sure to establish connection to the DB beforehand
    conn = sqlite3.connect('airport_coordinates.db')
    engine = create_engine('sqlite:///airport_coordinates.db')
    
    pd.DataFrame.to_sql(self = df, name = 'airportCoords', con=engine, if_exists='append', index=False,chunksize=100000)
    
    conn.commit()
    conn.close()  
"""    
    ##You don't need any of this below. You can take the df created from the csv import and 
    ##import that directly into the database
    
#    df = df[df.Country == 'United States']
#    iatas = df['IATA'].values
#    latitude = df['lat'].values
#    longitude = df['long'].values
#
#    iata = []
#    lats = []
#    longs = []
#    for row in df: 
#        iata.append(iatas)
#        lats.append(latitude)
#        longs.append(longitude)
#
#    conn = sqlite3.connect('airport_coordinates.db')
#    c = conn.cursor()
#
#    c.execute('INSERT INTO airport_coordinates (IATA, lat, long) VALUES (%r, %r, %r)' %(tuple(iata), tuple(lats), tuple(longs)))

def query_db(file, airline):

    engine = create_engine('sqlite:///airport_coordinates.db')
    conn = sqlite3.connect('airport_coordinates.db')

    airport_relpath = "data/processed/%s_%s_combined.csv" % (file, airline)
    script_dir = os.path.dirname(os.getcwd())
    temp_file = os.path.join(script_dir, airport_relpath)
    fields = ["Origin_Airport_Code", "Destination_Airport_Code"]

    labels = ['Origin', 'x', 'y']

    # stores airports from 2008 file
    df_airports = pd.read_csv(temp_file, usecols=fields)

    origin_airport_list = df_airports["Origin_Airport_Code"]
    destination_airport_list = df_airports["Destination_Airport_Code"]

    origin_airport_list = origin_airport_list.unique()
    destination_airport_list = destination_airport_list.unique()

    origin_sql = "SELECT * FROM airportCoords WHERE iata IN ('%s')" %("', '".join(origin_airport_list))
    destination_sql = "SELECT * FROM airportCoords WHERE iata IN ('%s')" %("', '".join(destination_airport_list))

    df_origin = pd.read_sql(origin_sql,engine)
    df_destination  = pd.read_sql(destination_sql,engine)

    lats_o = []
    longs_o = []
    lats_d = []
    longs_d = []

    for i in range(0, 81):
        for j in range(0, 81):
            if origin_airport_list[i] == df_origin.iloc[j].iata:
                lats_o.append(df_origin.iloc[j].lat)
                longs_o.append(df_origin.iloc[j].long)
    for i in range(0, 83):
        for j in range(0, 83):
            if destination_airport_list[i] == df_destination.iloc[j].iata:
                lats_d.append(df_destination.iloc[j].lat)
                longs_d.append(df_destination.iloc[j].long)
    
    df_o = pd.DataFrame({'iata':origin_airport_list, 'lat':lats_o, 'long':longs_o})
    df_l = pd.DataFrame({'iata':destination_airport_list, 'lat':lats_d, 'long':longs_d})
    return(df_o, df_l)

    #print(df_destination)

    """
    temp = Table(
    "airports_temp",
    metadata,
    *(Column(field, String) for field in fields),
    prefixes=['TEMPORARY']
    )
    with engine.begin() as conn:
        # insert CSV values into a temporary table in SQLite
        temp.create(conn, checkfirst=True)
        df_airports.to_sql(temp.name, engine, if_exists='append')

    filter = select([temp.c.Origin_Airport_Code]).distinct().subquery()
    sql = select([airport_coords.c.lat, airport_coords.c.long, airport_coords.c.iata]).where(airport_coords.c.iata.c.in_(subquery))
    df_origin = pd.read_sql(sql, engine)
    print(df_origin)

    # SQLAlchemy table wrangling
    


    metadata = MetaData()
    airport_coords = Table('airportCoords', metadata, autoload=True, autoload_with=engine)
    temp = Table(
        "airports_temp",
        metadata,
        *(Column(field, String) for field in fields),
        prefixes=['TEMPORARY']
    )
    with engine.begin() as conn:
        # insert CSV values into a temporary table in SQLite
        temp.create(conn, checkfirst=True)
        df_airports.to_sql(temp.name, engine, if_exists='append')

    # Join the airport coords against the temporary table
    joined = airport_coords.join(airports_temp, airport_coords.c.iata==temp.c.Origin_Airport_Code)
 
    # select coordinates per airport, include the iata code
    sql = select([airport_coords.c.lat.values, airport_coords.c.long.values, airport_coords.c.iata.values]).select_from(joined)
    df_origin = pd.read_sql(sql.distinct(), engine)

"""
    """
    origin = df_airports.Origin_Airport_Code.values
    dest = df_airports.Destination_Airport_Code.values
    sql = ('SELECT * FROM airportCoords WHERE iata IN %s' %(origin))

    df_origin = pd.read_sql(sql, engine)
    """
    """
    """

def Remove_Frequency(file, airline, include_data, can_limit, zs_limit):
    script_dir = os.path.dirname(os.getcwd())
    comb_path = "data/processed/%s_%s_combined.csv" % (file,airline)

    df_net_tuple = pd.DataFrame()
    comb_file = os.path.join(script_dir, comb_path)
    fields = ["Origin_Airport_Code", "Destination_Airport_Code"]
    df_net = pd.read_csv(comb_file, usecols=fields)

    df_net_tuple["Origin"] = df_net.Origin_Airport_Code
    df_net_tuple["Destination"] = df_net.Destination_Airport_Code

    # output_graph = []
    # output_cluster_size = []
    # output_cluster_frac = [] 
    # NoD = []

    graph = [tuple(x) for x in df_net_tuple.to_records(index=False)]
    G = nx.Graph()

    G.add_edges_from(graph)

    tempG = G.copy()
    #largest_component_b = max(nx.connected_components(G), key=len)
    #print(G.nodes())

    Airport_Dict = {}
    for i in G.nodes():
        Airport_Dict[i] = 0

    #print(Airport_Dict)


    Total_List = oc.Get_Remove_List(file,include_data, airline, can_limit, zs_limit)
    
    if int(file)%4 == 0:
        total_day = 366
    else:
        total_day = 365
        
    for j in range(total_day):
        #airport_list = df_RList.loc[j,'Impacted_Airports'] 
        airport_list = Total_List[j]
        #airport_list = ['ATL', 'BNA', 'BWI', 'DCA', 'EWR', 'FLL', 'HNL', 'LAX', 'MCO', 'MSP', 'OGG', 'SJC', 'SJU']
        #print(airport_list)
        #NoD.append(j)
        for l in airport_list:
            tempG.remove_node(l)
            #largest_component_b = max(nx.connected_components(tempG), key=len)

            Airport_Dict[l] = Airport_Dict[l] + 1



        tempG = G.copy()

    print(Airport_Dict)

def weighted_edge(file, airline):
    script_dir = os.path.dirname(os.getcwd())
    rel_path = "data/processed/%s_%s_combined.csv" % (file, airline)
    raw_file = os.path.join(script_dir, rel_path)
    fields = ["Destination_Airport_Code", "Origin_Airport_Code", "Can_Status"]
    df = pd.read_csv(raw_file, usecols=fields)
    by_origin = df.groupby([df.Origin_Airport_Code]).Can_Status.count()
    airport_list = by_origin.index.tolist()
    df = df[df['Destination_Airport_Code'].isin(airport_list)]
    #print(df)
    df_tuple = pd.DataFrame()
    df_weighted = df.groupby([df.Origin_Airport_Code, df.Destination_Airport_Code]).Can_Status.count().reset_index()
    df_tuple["Origin"] = df_weighted.Origin_Airport_Code
    df_tuple["Destination"] = df_weighted.Destination_Airport_Code
    file_str = int(str(file)[:4])
    if calendar.isleap(file_str) == 1:
        days = 366
    else:
        days = 365
    
    #print(df_weighted.Can_Status)
    # df_tuple["Weight"] = df_weighted.Can_Status/days
    # df_tuple.Weight = 1/df_tuple.Weight
    df_tuple["Weight"] = df_weighted.Can_Status

    weight_sum = 0
    for i in range(0, 464):
        weight_sum = weight_sum + df_tuple.iloc[i].Weight
    for i in range(0, 464):
        df_tuple.Weight.values[i] = (df_tuple.Weight.values[i] / weight_sum)*1000
        if df_tuple.Weight.values[i] == 0:
            df_tuple.Weight.values[i] = 0.1
    df_tuple.Weight = df_tuple.Weight.astype(float)



    return(df_tuple.Weight.values)


def map_plot(file, airline):
    df_o, df_d = query_db(file, airline)
    df_d = df_d.drop([81,82])

    edge_weight = weighted_edge(file, airline)

    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
    m.readshapefile('states', name='states', drawbounds='True')
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,60.))
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='white', lake_color='#46bcec')
    ##
    x_all = []
    y_all = []
   
    for i in range(0, 81):
        x_all.extend(df_o.iloc[i].long, df_d.iloc[i].long)
        y_all.extend(df_o.iloc[i].lat, df_d.iloc[i].lat)
    ##    
    x, y = m(x_all, y_all)
    #Create a variable that has the line weights (in order). Should be able to weight the color using plot tools.
    m.plot(x, y, color='blue') 

    node_o_long, node_o_lat = m(df_o.long.values, df_o.lat.values)
    m.scatter(node_o_long, node_o_lat, marker='.', color='red', zorder=5)
    # plot airports as points
    plt.show()

    """
    script_dir = os.path.dirname(os.getcwd())
    rel_path = "data/processed/%s.csv" % 'airport_data'
    raw_file = os.path.join(script_dir, rel_path)
    fields = ["IATA", "Country", "lat", "long"]
    # dataframe for United States airport lat/long coords
    df = pd.read_csv(raw_file, usecols=fields)
    df = df[df.Country == 'United States']
    latitude = df['lat'].values
    longitude = df['long'].values
    """
map_plot('2008', 'AA')