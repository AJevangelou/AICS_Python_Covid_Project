import math
import os

import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import geopandas
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------
#                                        GLOBAL VARIABLES DEFINITION
# -----------------------------------------------------------------------------------------------------------

COVID_PATH = os.path.join("datasets", "covid")
COVID_CONFIRMED = os.path.join("confirmed.csv")
COVID_DEATHS = os.path.join("deaths.csv")
COVID_RECOVERED = os.path.join("recovered.csv")


# -----------------------------------------------------------------------------------------------------------
#                                          FUNCTION DEFINITIONS:
# -----------------------------------------------------------------------------------------------------------

# Load csv files
def load_data(covid_data_path):
    csv_path = os.path.join(COVID_PATH, covid_data_path)
    return pd.read_csv(csv_path)


# Report unique countries
def report_countries(dataframe, groupedBy):
    groupedDF = dataframe.groupby([groupedBy])
    for country, country_df in groupedDF:
        print(country)


# Report the time span of the data
def report_date(dataframe):
    index = dataframe.columns
    print("The complete time span for the data is from: " + index[2] + " to: " + index[-1])


# Update dataframe by Country
def updateDF(dataframe: DataFrame, groupedBy):
    meanLatitude = []
    meanLongitude = []
    updatedDF: DataFrame

    groupedDF = dataframe.groupby([groupedBy])
    for country, country_df in groupedDF:
        meanLatitude.append(country_df.loc[:, "Lat"].mean())
        meanLongitude.append(country_df.loc[:, "Long"].mean())

    updatedDF = groupedDF.sum()
    updatedDF["Lat"] = meanLatitude
    updatedDF["Long"] = meanLongitude

    return updatedDF


# Plot to world map
def plotToMap(dataFrame, typeOfData, date, divisionConstant):
    # df_geo = dataFrame.GeoDataFrame(dataFrame, geometry=geopandas.points_from_xy(dataFrame["Lat"], dataFrame["Long"]))

    # From GeoPandas, our world map data
    worldmap = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color="lightgrey", ax=ax)

    plt.title("COVID-19 {} on {} \n".format(typeOfData, date))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # Creating axis limits and title
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])

    # Get sizes for the figure
    size = dataFrame[date].div(divisionConstant)

    plt.scatter(dataFrame["Long"], dataFrame["Lat"], c=size, s=size, cmap='plasma', edgecolors="black")
    cbar = plt.colorbar()
    cbar.set_label("Number of {} \n (Representation is divided by {})".format(typeOfData, divisionConstant))
    plt.tight_layout()
    plt.grid()
    plt.show()


# Get total cases
def sumCases(dataframe, typeOfdata):
    updatedDF: DataFrame
    dataframe[typeOfdata] = dataframe.iloc[:, 4:].sum(axis=1)
    updatedDF = dataframe
    return updatedDF


# Get top 10 countries by total cases
def topCases(dataframe, typeOfdata):
    topDF = dataframe.nlargest(10, typeOfdata)

    # Figure Size
    fig, ax = plt.subplots(figsize=(10, 5))

    # Horizontal Bar Plot
    ax.barh(topDF.index.values, topDF[typeOfdata])

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=2)

    # Add x, y gridlines
    ax.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=10, color='grey')


    # Add Plot Title
    ax.set_title('COVID-19 {} as of 29-May-21'.format(typeOfdata), loc='center')
    # Add x label
    ax.set_xlabel("{}".format(typeOfdata), loc='center')

    # Show Plot
    plt.show()


def main():
    # Load all data from the .csv
    df_covid_confirmed = load_data(COVID_CONFIRMED)
    df_covid_deaths = load_data(COVID_DEATHS)
    df_covid_recovered = load_data(COVID_RECOVERED)

    # Update DataFrames grouping them by Country and Giving them a Mean Lat and Long
    updated_confirmed = updateDF(df_covid_confirmed, "Country/Region")
    updated_deaths = updateDF(df_covid_deaths, "Country/Region")
    updated_recovered = updateDF(df_covid_recovered, "Country")

    # Saving DataFrames to new CSV files (Optionally)
    updated_recovered.to_csv('updated_recovered.csv')
    updated_deaths.to_csv('updated_deaths.csv')
    updated_confirmed.to_csv('updated_confirmed.csv')

    # Reporting countries and time span of data
    report_countries(updated_confirmed, "Country/Region")
    report_date(updated_confirmed)

    # Plotting confirmed cases
    plotToMap(updated_confirmed, "Confirmed Cases", "1/22/20", 1)
    plotToMap(updated_confirmed, "Confirmed Cases", "5/28/21", 100000)

    # Plotting recovered
    plotToMap(updated_recovered, "Recovered", "1/22/20", 1)
    plotToMap(updated_recovered, "Recovered", "5/28/21", 10000)

    # Plotting Deaths
    plotToMap(updated_deaths, "Deaths", "1/22/20", 1)
    plotToMap(updated_deaths, "Deaths", "5/28/21", 1000)

    # Sum cases and save updated CSV files
    updated_confirmed = sumCases(updated_confirmed, "Total Confirmed")
    updated_confirmed.to_csv('updated_confirmed.csv')

    updated_recovered = sumCases(updated_recovered, "Total Recovered")
    updated_recovered.to_csv('updated_recovered.csv')

    updated_deaths = sumCases(updated_deaths, "Total Deaths")
    updated_deaths.to_csv('updated_deaths.csv')

    # Get top 10 countries by total cases
    topCases(updated_confirmed, "Total Confirmed")
    #Get top 10 countries for recovered cases
    topCases(updated_recovered, "Total Recovered")
    # Get top 10 countries for deaths
    topCases(updated_deaths, "Total Deaths")

if __name__ == "__main__":
    main()
