import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


class Internet:
    "Class for preparing the internet user data for visualisation in tableau"

    def __init__(self, file_name) -> None:
        self.df_raw = pd.read_csv(file_name)
        self.df_cleaned = self.clean(self.df_raw)
        self.df_baltics = self.baltics_eu(self.df_cleaned)
        self.export(self.df_baltics, "internet_stats.csv")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unused columns, rename columns and rename data"""
        df = df.drop(
            ["DATAFLOW", "LAST UPDATE", "freq", "unit", "ind_type", "OBS_FLAG"], axis=1
        )
        df = df.rename(
            columns={
                "geo": "Country",
                "indic_is": "Indicator",
                "OBS_VALUE": "Percent of Total",
                "TIME_PERIOD": "Year",
            }
        )
        indicator = {
            "I_IHIF": "Seeking health information",
            "I_IUBK": "Internet banking",
            "I_IUEM": "Sending/receiving e-mails",
            "I_IUIF": "Finding information about goods and services",
            "I_IUOLM": "Online learning material",
            "I_IUPH1": "Telephoning or video calls",
            "I_IUSELL": "Selling goods or services",
            "I_IUSNET": "Participating in social networks",
        }
        country = {
            "EU27_2020": "EU",
        }

        df["Indicator"] = df["Indicator"].map(indicator)
        df["Country"] = df["Country"].map(country)
        return df

    def baltics_eu(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter only EU data"""
        df = df[df["Country"].isin(["EU"])].copy()
        return df

    def export(self, data: pd.DataFrame, file_name: str) -> None:
        """Export file to csv"""
        data.to_csv(file_name)


class Mobility:
    """
    Class for working with the mobility data from google
    """

    def __init__(self, file_name) -> None:
        self.df_raw = pd.read_csv(file_name)
        self.df_cleaned = self.clean(self.df_raw)
        self.country = "Latvia"
        self.country_data = self.filter_country(self.df_cleaned, self.country)
        self.workplace_data = self.workplaces(self.country_data)
        self.run()

    def remove_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter only country-wide data"""
        country_mask = df["sub_region_1"].isna()
        df = df[country_mask]
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unused columns, execute remove-regions method, convert date column to datetime"""
        df = self.remove_regions(df)
        df = df.drop(
            [
                "sub_region_1",
                "sub_region_2",
                "metro_area",
                "iso_3166_2_code",
                "census_fips_code",
                "place_id",
            ],
            axis=1,
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    def export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Export file to csv"""
        df.to_csv("final.csv", index=False)

    def filter_country(self, df: pd.DataFrame, country: str):
        """Filter data by specific country"""
        df = df[df["country_region"] == country].copy()
        return df

    def workplaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select data for workplace activity only, rename column and reset index"""
        df = df[["country_region", "date", "workplaces_percent_change_from_baseline"]]

        df = df.rename(
            columns={
                "workplaces_percent_change_from_baseline": "Percent_change",
            }
        )
        df.reset_index(inplace=True)
        return df

    def plot(self, df: pd.DataFrame) -> None:
        """Plot graph for % change in indicator over time compared to baseline"""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.date, df.Percent_change)
        ax.set_ylabel("% change from baseline")
        plt.show()

    def investigate_spikes(self, df: pd.DataFrame) -> None:
        print(
            """After investigation days with activity higher than 30% above baseline were transfered workdays 
            and those below -50% were holidays with the dates below:"""
        )
        print(df[df.Percent_change > 30])
        print(df[df.Percent_change < -50])

    def clean_holiday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace holiday data with average of the previous and next workday"""
        # Create a list of holidays
        holidays = df[df.Percent_change < -50].date.values
        # Convert holidays list to datetime
        holidays = pd.to_datetime(holidays)
        # Initialize an empty list to store the averages
        averages = []
        # Iterate through the holidays
        for holiday in holidays:
            next_bd = holiday
            prev_bd = holiday
            while (next_bd in holidays) | (next_bd.isoweekday in (6, 7)):
                next_bd = next_bd + dt.timedelta(1)
            while (prev_bd in holidays) | (next_bd.isoweekday in (6, 7)):
                prev_bd = prev_bd - dt.timedelta(1)
            # Select the value of the previous business day
            prev_value = df.loc[df["date"] == next_bd, "Percent_change"].values[0]
            # Select the value of the next business day
            next_value = df.loc[df["date"] == prev_bd, "Percent_change"].values[0]
            # Calculate the average of the previous and next business days
            average = (prev_value + next_value) / 2
            # Append the average to the averages list
            averages.append(average)
        # Iterate through the holidays and their corresponding averages
        for holiday, avg in zip(holidays, averages):
            # Change the value of the holiday to the average
            df.loc[df["date"] == holiday, "Percent_change"] = avg
        return df

    def clean_transfered_workday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace transfered workday data with the following Sunday data"""
        df["Next_day_value"] = df["Percent_change"].shift(1)
        df[df.Percent_change > 30].Percent_change = df["Next_day_value"]
        df["Percent_change"] = df.apply(
            lambda x: x["Next_day_value"]
            if x["Percent_change"] > 30
            else x["Percent_change"],
            axis=1,
        )
        return df

    def run(self):
        self.plot(self.workplace_data)
        self.investigate_spikes(self.workplace_data)
        df = self.clean_transfered_workday(self.workplace_data)
        df = self.clean_holiday(df)
        self.plot(df)


class Time_series_model:
    "Used to model activity in workplaces in Latvia using SARIMA"

    def __init__(self, data: pd.DataFrame) -> None:
        self.order = (2, 0, 1)  # (p, d, q)
        self.seasonal_order = (1, 1, 0, 52)  # (P, D, Q, S)
        self.days = 120
        self.run(data)

    def weekly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create weekly data"""
        data = data.groupby(pd.Grouper(key="date", freq="W")).mean()
        return data

    def ADF_test(self, data: pd.DataFrame) -> None:
        """Use to test stationarity"""
        ad_fuller_result = adfuller(data)
        print(f"ADF Statistic: {ad_fuller_result[0]}")
        print(f"p-value: {ad_fuller_result[1]}")

    def auto_sarima(self, weekly):
        """use to find optimal parameters for SARIMA"""
        arima_model = auto_arima(
            weekly,
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            max_d=2,
            m=52,
            d=0,
            seasonal=True,
            start_P=0,
            start_Q=0,
            max_P=5,
            max_D=5,
            D=1,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

    def model(self, data: pd.DataFrame) -> SARIMAX:
        """create model"""
        model = SARIMAX(
            data[:-28], order=self.order, seasonal_order=self.seasonal_order
        )
        return model

    def fit(self, model: SARIMAX) -> SARIMAX:
        """fit model"""
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

    def predict(self, model_fit: SARIMAX, data: pd.DataFrame, days: int) -> np.array:
        """Predict 2023 and plot the fitted values compare to actual"""
        predictions = model_fit.predict(start=len(data[:-28]), end=len(data) + 52)
        forecast = model_fit.fittedvalues
        forecast = forecast.append(predictions)
        plt.figure(figsize=(15, 7.5))
        plt.plot(forecast[60:], color="r", label="model")
        plt.plot(data["Percent_change"], color="y", label="actual")
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel("% change from baseline", fontsize=14)
        plt.show()

    def diagnose(self, model_fit: SARIMAX) -> None:
        """Plot diagnostic plots"""
        model_fit.plot_diagnostics(figsize=(16, 8))
        plt.show()

    def run(self, data: pd.DataFrame) -> None:
        weekly = self.weekly(data)
        self.ADF_test(weekly)
        model = self.model(weekly)
        fitted = self.fit(model)
        self.predict(fitted, weekly, self.days)
        self.diagnose(fitted)


class PCA_countries:
    """
    Perform PCA and create a biplot on google mobility data
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.ready_data_df = self.ready_data(self.data)
        self.pca, self.pca_fit = self.perform_pca(self.ready_data_df)
        self.scree_plot(self.pca)
        self.biplot(
            self.pca_fit,
            np.transpose(self.pca.components_),
            1,
            2,
            self.ready_data_df.columns[0:],
        )

    def ready_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """drop unused columns, reindex, calculate the mean of data by country from 15.09.2022 to 15.10.2022"""
        data.index = np.arange(1, len(data) + 1)
        data = data.drop(
            ["country_region", "parks_percent_change_from_baseline"], axis=1
        )
        data = data[data["date"] > "2022-09-15"].groupby("country_region_code").mean()
        data = data.dropna()
        data = data[~(data > 50).any(1)]
        return data

    def perform_pca(self, data: pd.DataFrame) -> Union[pd.DataFrame, PCA]:
        pca = PCA(n_components=5)
        pca_components = pca.fit_transform(data)
        return pca, pca_components

    def scree_plot(self, pca: PCA) -> None:
        PC_values = np.arange(pca.n_components_) + 1
        plt.plot(
            PC_values, pca.explained_variance_ratio_, "o-", linewidth=2, color="blue"
        )
        plt.title("Scree Plot")
        plt.xlabel("Principal Component", fontsize=14)
        plt.ylabel("Variance Explained", fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.show()

    def biplot(self, score, coeff, pcax: int, pcay: int, labels: list = None) -> None:
        pca1 = pcax - 1
        pca2 = pcay - 1
        xs = score[:, pca1]
        ys = score[:, pca2]
        n = score.shape[1]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        plt.scatter(xs * scalex, ys * scaley)
        for j, txt in enumerate(self.ready_data_df.index):
            plt.annotate(txt, (xs[j] * scalex, ys[j] * scaley))
        for i in range(n):
            plt.arrow(0, 0, coeff[i, pca1], coeff[i, pca2], color="r", alpha=0.5)
            if labels is None:
                plt.text(
                    coeff[i, pca1] * 1.15,
                    coeff[i, pca2] * 1.15,
                    "Var" + str(i + 1),
                    color="g",
                    ha="center",
                    va="center",
                )
            else:
                plt.text(
                    coeff[i, pca1] * 1.15,
                    coeff[i, pca2] * 1.15,
                    labels[i],
                    color="g",
                    ha="center",
                    va="center",
                )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel("PC{}".format(pcax))
        plt.ylabel("PC{}".format(pcay))
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.grid()


class Household:
    """
    Prepare household expenditure data for visualisation in Tableau
    """

    def __init__(self, file_name: str) -> None:
        self.data = pd.read_excel(file_name, sheet_name="all")
        self.cleaned = self.transform(self.data)
        self.export(self.cleaned, "Household_expenditure.csv")
        self.export(self.compare_19_22_eu(self.cleaned), "HE_EU.csv")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """drop unused cols, unpivot data into measure and value columns"""
        df = df.drop(["Total"], axis=1)
        df = df.melt(
            id_vars=["country", "year"], var_name="Measure", value_name="Value"
        )
        return df

    def compare_19_22_eu(self, df: pd.DataFrame) -> pd.DataFrame:
        """compare 2019 and 2022 data by indicator"""
        df_eu = df[df["country"] == "European Union - 27 countries (from 2020)"]
        df_19 = df_eu[df_eu["year"] == 2019]
        df_21 = df_eu[df_eu["year"] == 2021]
        df_combined = pd.merge(df_21, df_19, how="inner", on=["country", "Measure"])
        df_combined["Change"] = (
            (df_combined["Value_x"] - df_combined["Value_y"])
            / df_combined["Value_y"]
            * 100
        )
        return df_combined

    def export(self, data: pd.DataFrame, file_name: str) -> None:
        """export to csv"""
        data.to_csv(file_name, index=False)


if __name__ == "__main__":
    mobility = Mobility("Global_Mobility_Report.csv")
    sarima = Time_series_model(mobility.workplace_data[["Percent_change", "date"]])
    household = Household("HE.xlsx")
    internet = Internet("internet_activity.csv")
    pca = PCA_countries(mobility.df_cleaned)
