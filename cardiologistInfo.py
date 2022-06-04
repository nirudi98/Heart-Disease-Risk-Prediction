import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim
import pandas as pd


def ShowingResult():
    csv_filename = 'Cardiologist.csv'
    file_path = 'D:/NSBM/NSBM/year 4/research/heart disease prediction/'
    df = pd.read_csv(file_path + csv_filename)
    print(df)
    df = df.style.set_caption("").applymap(lambda x: 'background-color : #C3B1E1')\
        .set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '#AA98A9')]},
                           {'selector': 'th', 'props': [('background-color', '#E6E6FA')]}])
    return df


def hospital_map():
    hospitals = pd.read_csv('D:/NSBM/NSBM/year 4/research/heart disease prediction/hospital_locations.csv')
    # view the dataset
    print(hospitals.head())
    country = [7.8731, 80.7718]
    sl_map = folium.Map(location=country, zoom_start=8)
    for index, hospitals in hospitals.iterrows():
        location = [hospitals['Latitude'], hospitals['Longitude']]
        folium.Marker(location, popup=f'Hospital Name:{hospitals["Hospital"]}\n Cardiologist:{hospitals["Name"]}').add_to(
            sl_map)

    # save map to html file
    sl_map.save('templates/map.html')


if __name__ == '__main__':
    hospital_map()