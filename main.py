import math
import os

import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import geopandas
import matplotlib.pyplot as plt
from PIL import Image
import glob

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


# Update dataframe by Country and return mean Longitude and Latitude for each country
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


def plotToMapThreshold(dataFrame, typeOfData, date, divisionConstant, threshold, description):

    # From GeoPandas, our world map data
    worldmap = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color="lightgrey", ax=ax)

    # Create title, xLabel, yLabel, set x and y axis limits
    plt.title("COVID-19 {} on {} \nCountries with 100+ cases are represented with red".format(typeOfData, date))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Creating axis limits
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])

    # Filter (drop) the countries with more or equal than 100 cases
    lessData = dataFrame.drop(dataFrame.index[dataFrame[date] < threshold])

    dataFrame.reset_index()

    # Filter (drop) the countries with less than 100 cases
    dataFrame = dataFrame.drop(dataFrame.index[dataFrame[date] >= threshold])

    size = dataFrame[date]

    # Divide by a number, because size of circles would be too big to show
    lessSize = lessData[date].div(divisionConstant)

    # Add scatter plots to figure. Red for >= cases
    plt.scatter(dataFrame["Long"], dataFrame["Lat"], s=size, color='blue', edgecolors="black", alpha=0.5)
    plt.scatter(lessData["Long"], lessData["Lat"], s=lessSize, color='red', edgecolors="black", alpha=0.5)

    # Adding tight layout and grid
    plt.tight_layout()
    plt.grid()

    # Saving the figure
    path = os.path.join("./datasets/figures/", typeOfData.replace(" ", ""))
    os.makedirs(path, exist_ok=True)
    plt.savefig("./datasets/figures/{}/".format(typeOfData.replace(" ", "")) + typeOfData + description)

    # Displaying figure
    plt.show()


# Get total cases
def sumCases(dataframe, typeOfdata):
    updatedDF: DataFrame
    dataframe[typeOfdata] = dataframe.iloc[:, 4:].sum(axis=1)
    updatedDF = dataframe
    return updatedDF


# Get top 10 countries by total cases
def topCases(dataframe, typeOfdata, description):
    topDF = dataframe.nlargest(10, typeOfdata)
    COUNTRIES = topDF.index.tolist()

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
    createHistogram(dataframe, COUNTRIES, description)


# Create a gif from saved figures
def createGif():
    # Build GIF
    # Create the frames
    frames = []
    imgs = glob.glob("./datasets/figures/*.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save the png images into a GIF file that loops forever
    frames[0].save('png_to_gif.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=300, loop=0)

# Create histogram for the top 10 Countries
def createHistogram(dataFrame, countries_array, description):
    # Get all dates from pandas
    list_of_dates = dataFrame.columns.values.tolist()[2:]

    len_of_dates = len(list_of_dates)

    new_cases = [[], [], [], [], [], [], [], [], [], []]
    for i in range(0, len(countries_array)):
        list_of_values = dataFrame.loc[countries_array[i]][2:].values
        print(countries_array[i])

        new_cases[i].append(list_of_values[0])
        for j in range(1, len(list_of_values)):
            x = list_of_values[j] - list_of_values[j - 1]
            if x < 0:
                new_cases[i].append(0)
            else:
                new_cases[i].append(x)

    # Create figures and axis for the plots
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("COVID-19 {} - Top 10 Country/Region".format(description))
    fig.supxlabel("As of 29-May-2021")
    fig.supylabel("New Cases")

    #Iterate through top 10 countries and add subplots to figures
    index = 0
    while index < 10:
        for row in range(0, 3):
            for column in range(0, 4):
                if index >= 10:
                    break
                else:
                    axes[row, column].plot(list_of_dates, new_cases[index])
                    axes[row, column].set_title(countries_array[index])
                    axes[row, column].set_xticks([0, int(len_of_dates / 3), int(2*len_of_dates/3), int(len_of_dates)])
                    axes[row, column].xaxis.set_ticklabels(["Jan 20", "Jul 20", "Jan 21", "Jul 21"])
                    index += 1

    # Delete the figures which are not used
    fig.delaxes(axes[2][2])
    fig.delaxes(axes[2][3])

    # Display figures
    plt.tight_layout()
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

    # Plotting confirmed cases with threshold of 100 cases and dividing to make them easier to view
    plotToMapThreshold(updated_confirmed, "Confirmed Cases", "1/22/20", 1, 100, "2")
    plotToMapThreshold(updated_confirmed, "Confirmed Cases", "5/29/21", 100000, 100, "1")

    # Plotting recovered cases with threshold of 100 cases and dividing to make them easier to view
    plotToMapThreshold(updated_recovered, "Recovered", "1/22/20", 1, 100, "2")
    plotToMapThreshold(updated_recovered, "Recovered", "5/29/21", 10000, 100, "1")

    # Plotting Deaths cases with threshold of 100 cases and dividing to make them easier to view
    plotToMapThreshold(updated_deaths, "Deaths", "1/22/20", 1, 100, "2")
    plotToMapThreshold(updated_deaths, "Deaths", "5/29/21", 1000, 100, "1")

    # Get top 10 countries by total cases
    topCases(updated_confirmed, "5/29/21", "Confirmed Cases")
    # Get top 10 countries for recovered cases
    topCases(updated_recovered, "5/29/21", "Recovered Cases")
    # Get top 10 countries for deaths
    topCases(updated_deaths, "5/29/21", "Deaths")

    # Create an animated GIF with
    createGif()


if __name__ == "__main__":
    main()
