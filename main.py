import os

import numpy as np
import pandas as pd
from pandas import DataFrame

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

def load_data(covid_data_path):
    csv_path = os.path.join(COVID_PATH, covid_data_path)
    return pd.read_csv(csv_path)


def report_countries(dataframe, groupedBy):
    groupedDF = dataframe.groupby([groupedBy])
    for country, country_df in groupedDF:
        print(country)

def report_date(dataframe):
    index = dataframe.columns
    print("The complete time span for the data is from: " + index[2] + " to: " + index[-1])

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

    #Reporting countries and time span of data
    report_countries(updated_confirmed, "Country/Region")
    report_date(updated_confirmed)
if __name__ == "__main__":
    main()
