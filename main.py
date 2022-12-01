import os

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


def report_attributes(data):
    for col_name in data.columns:
        print("\nReporting value counts information on {}".format(col_name))
        data_col = data[col_name]
        data_col_series = data_col.value_counts()
        print(data_col_series)


def createNewDf(df_confirmed, df_deaths, df_recovered, df_by_countries):
    # Create arrays to populate the new DataFrame for Pandas
    countriesArray = []
    meanLatitude = []
    meanLongitude = []
    totalCases = []
    totalDeaths = []
    totalRecovered = []

    # Group confirmed cases by Country and iterate through them
    confirmed = df_confirmed.groupby("Country/Region")

    for country, country_df in confirmed:
        #print(country)
        countriesArray.append(country)

        # Get average latitude for each country
        #print("Mean latitude is: " + str(country_df.loc[:, "Lat"].mean()))
        meanLatitude.append(country_df.loc[:, "Lat"].mean())

        # Get average longitude for each country
        #print("Mean longitude is: " + str(country_df.loc[:, "Long"].mean()))
        meanLongitude.append(country_df.loc[:, "Long"].mean())

        # Sum total cases (keeping only numeric values)
        country_total = country_df.iloc[:, 4:].sum(1, numeric_only=True)

        # Check if we have to iterate through regions
        if country_df.shape[0] > 1:
            #print("Total cases for the country: " + str(country_total.values.sum()) + "\n")
            totalCases.append(country_total.values.sum())
        else:
            #print("Total cases for the country: " + str(country_total.values) + "\n")
            totalCases.append(country_total.values)

    #Start populating a new DataFrame with rows and columns
    df_by_countries["Country"] = countriesArray
    df_by_countries["Mean Latitude"] = meanLatitude
    df_by_countries["Mean Longitude"] = meanLongitude
    df_by_countries["Total Cases"] = totalCases

    # Group confirmed cases by Country and iterate through them
    deaths = df_deaths.groupby("Country/Region")

    # Iterate through countries to sum deaths
    for country, country_df in deaths:
        country_total_deaths = country_df.iloc[:, 4:].sum(1, numeric_only=True)

        if country_df.shape[0] > 1:

            #print("Total deaths for the country: " + str(country_total_deaths.values.sum()) + "\n")
            df_by_countries["Total Deaths"] = country_total_deaths.values.sum()
            totalDeaths.append(country_total_deaths.values.sum())

        else:
            #print("Total deaths for the country: " + str(country_total_deaths.values))
            totalDeaths.append(country_total_deaths.values)

    # Populate the new DataFrame with Total Deaths
    df_by_countries["Total Deaths"] = totalDeaths

    # Group recovered cases by Country and iterate through them
    recovered = df_recovered.groupby("Country")
    for country, country_df in recovered:
        country_total_recovered = country_df.iloc[:, 4:].sum(1, numeric_only=True)
        if country_df.shape[0] > 1:
            #print("Total recovered for the country: " + str(country_total_recovered.values.sum()) + "\n")
            df_by_countries["Total Recovered"] = country_total_recovered.values.sum()
            totalRecovered.append(country_total_recovered.values.sum())
        else:
            #print("Total recovered for the country: " + str(country_total_recovered.values))
            totalRecovered.append(country_total_recovered.values)
    # Populate the new DataFrame with Total Recovered
    df_by_countries["Total Recovered"] = totalRecovered
    print(countriesArray)
    return df_by_countries

def getDates(dataFrame):
    print("The time span of the given dataset is between those dates: " + str(dataFrame.columns.values[4]) + " and " + str(dataFrame.columns.values[-1]))


def main():
    #Load all data from the .csv
    df_covid_confirmed = load_data(COVID_CONFIRMED)
    df_covid_deaths = load_data(COVID_DEATHS)
    df_covid_recovered = load_data(COVID_RECOVERED)

    # Create a new DataFrame to store the new values
    df_by_countries: DataFrame = pd.DataFrame()
    updatedDF = createNewDf(df_covid_confirmed, df_covid_deaths, df_covid_recovered, df_by_countries)

    # Save the new DataFrame to a .csv File
    updatedDF.to_csv('newDF')

    # Show the time span of the dataset
    getDates(df_covid_confirmed)

if __name__ == "__main__":
    main()
