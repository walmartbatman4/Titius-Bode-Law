import pandas as pd
import numpy as np

class regression:
    def dataHandler(self, planet_data):
        results = []
        for star in planet_data:
            x = [i + 1 for i in range(star[1])]
            y = [np.log(planet[1]) for planet in star[2]]
            slope, intercept = regression.linear_regression(x, y)
            Rsq = regression.squareR(x,y,slope,intercept)

        return results
    @staticmethod
    def squareR(x, y, slope, intercept):
        # Predicted values
        y_pred = [slope * xi + intercept for xi in x]

        # Mean of observed values
        y_mean = sum(y) / len(y)

        # Total sum of squares
        ss_total = sum((yi - y_mean) ** 2 for yi in y)

        # Residual sum of squares
        ss_res = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, y_pred))

        # R^2 calculation
        r2 = 1 - (ss_res / ss_total)
        return r2

    @staticmethod
    def linear_regression(x, y):
        lr = 0.01
        epochs = 1000

        x = np.array(x)
        y = np.array(y)

        m = 0.0  # slope
        b = 0.0  # intercept
        n = len(x)

        for _ in range(epochs):
            y_pred = m * x + b
            # Gradients
            dm = (-2 / n) * np.sum(x * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)
            # Update
            m -= lr * dm
            b -= lr * db

        return m, b
class dataExtractor:
    @classmethod
    def dataplanet(self):
        df = pd.read_csv("planets.csv", comment="#")
        # pl_name,hostname,sy_snum,sy_pnum,soltype,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbperlim
        sl = pd.read_csv("solar_system_planets.csv")
        df = df.sort_values(by=["sy_pnum", "hostname", "pl_orbper"])

        planet_array = []
        for star, group in df.groupby("hostname"):
            num_planets = int(group["sy_pnum"].iloc[0])
            planets = [
                [row["pl_name"], row["pl_orbper"]]
                for _, row in group.iterrows()
            ]
            planet_array.append([star,num_planets, planets])

        sl = sl.sort_values(by="pl_orbper")

        planets = [
            [row["pl_name"], row["pl_orbper"]]
            for _, row in sl.iterrows()
        ]

        planet_array.append(["Sun", len(planets), planets])
        return planet_array

Orbital_data = dataExtractor.dataplanet()

