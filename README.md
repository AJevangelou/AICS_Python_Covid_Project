# COVID-19 Data Analysis Project

This project involves analyzing COVID-19 data to visualize and understand the spread and impact of the virus across different countries. The analysis includes data preparation, visualization, and generating insights from the COVID-19 dataset.

## Table of Contents

- [Overview](#overview)
- [Data Source](#data-source)
- [Setup Instructions](#setup-instructions)
- [Analysis Steps](#analysis-steps)
  - [Loading Data](#loading-data)
  - [Updating DataFrames](#updating-dataframes)
  - [Printing DataFrame](#printing-dataframe)
  - [Plotting Data](#plotting-data)
  - [Generating GIFs](#generating-gifs)
  - [Top Countries Bar Plot](#top-countries-bar-plot)
  - [Wave Detection](#wave-detection)
- [Contributors](#contributors)

## Overview

This project was developed as part of the Advanced Python Programming course. It includes data download, preprocessing, and visualization to analyze the spread of COVID-19 using various techniques.

## Data Source

The dataset is obtained from Kaggle: [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)

Required files:
- `time_series_covid_19_confirmed.csv`
- `time_series_covid_19_deaths.csv`
- `time_series_covid_19_recovered.csv`

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/covid19-data-analysis.git
   cd covid19-data-analysis
   
2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows use env\Scripts\activate
3. **Install the required packages:**
    ```bash
   pip install -r requirements.text
4. Download the dataset and place it in the `dataset/covid` directory.

## Analysis Steps
### Loading Data
The load_data function loads the CSV files into Pandas DataFrames. Each file represents different types of COVID-19 
data (confirmed cases, deaths, and recoveries).

### Updating DataFrames
The updateDF function groups the data by country and calculates the mean latitude and longitude for each country.

### Printing DataFrame
The DataFrame is printed to verify the structure and content, focusing on the date columns to identify the 
range of available data.

### Plotting Data
The plot_data function generates scatter plots of COVID-19 data on a world map, distinguishing between data points 
with fewer than 100 cases and those with more.

### Generating GIFs
Saved plots are combined into GIFs to visualize the progression of COVID-19 cases, deaths, and recoveries over time.

### Top Countries Bar Plot
A horizontal bar plot is generated to display countries with the highest cumulative COVID-19 cases.

### Wave Detection
An algorithm is implemented to detect the waves of COVID-19 by analyzing daily case increases and decreases, 
smoothing data using a moving average to identify wave start and end dates.

