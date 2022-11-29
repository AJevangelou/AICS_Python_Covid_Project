import os

import pandas as pd

COVID_PATH = os.path.join("datasets", "covid")
print(COVID_PATH)
COVID_CONFIRMED = os.path.join("confirmed.csv")
COVID_DEATHS = os.path.join("time_series_covid_19_deaths.csv")
COVID_RECOVERED = os.path.join("time_series_covid_19_recovered.csv")


def load_data(covid_data_path):
    csv_path = os.path.join(COVID_PATH, covid_data_path)
    return pd.read_csv(csv_path)


def report_attributes(data):
    for col_name in data.columns:
        print("\nReporting value counts information on {}".format(col_name))
        data_col = data[col_name]
        data_col_series = data_col.value_counts()
        print(data_col_series)


def main():
    df_covid_confirmed = load_data(COVID_CONFIRMED)

    g = df_covid_confirmed.groupby("Country/Region")

    for country, country_df in g:
        print(country)
        print(country_df)

        # Get average latitude for each country
        print(country_df.loc[:, "Lat"].mean())

        # Get average longtitude for each country
        print(country_df.loc[:, "Long"].mean())

        # Sum total cases
        country_total = country_df.iloc[:len(country_df.index)].sum(axis=1, skipna=True, numeric_only=True)
        print(country_total.values)


if __name__ == '__main__':
    main()
