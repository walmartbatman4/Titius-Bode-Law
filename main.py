import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class regression:
    def __init__(self,planet_data):
        self.planet_data = planet_data
    def dataHandler(self):
        results = []
        for star in self.planet_data:
            #print(star[0])
            x = [i + 1 for i in range(star[1])]
            y = [np.log(planet[1]) for planet in star[2]]
            slope, intercept = regression.linear_regression(x, y)
            Rsq = regression.squareR(x,y,slope,intercept)
            mae = regression.MAE_log(x,y,slope,intercept)
            #output.temp(slope,intercept,x,y,star[0])
            results.append([star[0],star[1],slope,intercept,Rsq,mae])
        self.observation = results
        return results
    def recalc_with_correction(self):
        updated_results = []
        for i in range(len(self.observation)):
            R2 = self.observation[i][4]
            if R2 > 0.95:
                continue
            o1 = self.observation[i]
            s1 = self.planet_data[i]
            x = [j+1 for j in range(s1[1])]
            y = [np.log(planet[1]) for planet in s1[2]]
            '''y_pred = [o1[2] * xi + o1[3] for xi in x]

            errors = [abs(yi - ypi) for yi, ypi in zip(y, y_pred)]
            max_idx = np.argmax(errors)

            new_x = x + [x[max_idx]]
            new_y = y + [y_pred[max_idx] + (y[max_idx] - y_pred[max_idx]) * 0.5]'''
            errors = [y[i+1]-y[i] for i in range(len(y)-1)]
            max_idx = np.argmax(errors)

            new_x = [j+1 for j in range(s1[1]+1)]
            new_y = y.copy()
            new_y.insert(max_idx+1,(y[max_idx]+y[max_idx+1])*0.5)

            slope_new, intercept_new = regression.linear_regression(new_x, new_y)
            time_period = np.exp(slope_new*x[max_idx+1] + intercept_new)

            new_R2 = regression.squareR(new_x,new_y,slope_new,intercept_new)
            new_MAE = regression.MAE_log(new_x,new_y,slope_new,intercept_new)

            updated_results.append([s1[0],  new_x[max_idx+1],o1[4] ,new_R2, o1[5], new_MAE,time_period])

        return updated_results

    @staticmethod
    def squareR(x, y, slope, intercept):
        y_pred = [slope * xi + intercept for xi in x]
        y_mean = sum(y) / len(y)
        ss_total = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, y_pred))
        r2 = 1 - (ss_res / ss_total)
        return r2
    @staticmethod
    def MAE_log(x, y, slope, intercept):
        y_pred = [slope * xi + intercept for xi in x]
        abs_errors = [abs(yi - y_hat) for yi, y_hat in zip(y, y_pred)]
        return sum(abs_errors) /len(y)
    @staticmethod
    def linear_regression(x, y):
        lr = 0.01
        epochs = 1000
        #lam = 0.1

        x = np.array(x)
        y = np.array(y)
        #print(len(x))
        #print(len(y))
        m = 0.0
        b = 0.0
        n = len(x)

        for _ in range(epochs):
            y_pred = m * x + b
            #error = y_pred - np.array(y)
            #dm = (2 / n) * (np.dot(error, x)) + (2 * lam / n) * m
            dm = (-2 / n) * np.sum(x * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)
            m -= lr * dm
            b -= lr * db

        return m, b
class dataExtractor:
    @classmethod
    def dataplanet(self):
        df = pd.read_csv("planets.csv", comment="#")
        age = pd.read_csv("star_ages.csv")
        age = age.sort_values(by=["num_planets","star_name"])
        a = dict(zip(age["star_name"], age["age_gyr"]))
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
            planet_array.append([star,num_planets, planets,a[star]])

        sl = sl.sort_values(by="pl_orbper")
        planets = [
            [row["pl_name"], row["pl_orbper"]]
            for _, row in sl.iterrows()
        ]
        planet_array.insert(0,["Sun", len(planets), planets,age["age_gyr"][0]])
        planet_array.sort(key=lambda x: (-x[1], x[0]))
        return planet_array
class output:
    def data1(self,data):
        for star in data:
            print(star)

    def data2(self,data):
        bins = 100  # number of bins between 0 and 1
        plt.hist([star[4] for star in data], bins=bins, range=(0, 1), alpha=1, color="blue", label="R²")
        plt.hist([star[5] for star in data], bins=bins, range=(0, 1), alpha=1, color="red", label="MAE")

        plt.xlabel("Value")
        plt.ylabel("Number of Systems")
        plt.title("Distribution of R² and MAE across Systems")
        plt.legend()
        plt.show()

    def data3(self,planet_data,observation):
        ages = [row[3] for row in planet_data]  # assuming age is at 4th position
        r2_values = [row[4] for row in observation]  # R² is at 2nd position

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(ages, r2_values, color="blue", alpha=1, s=10)

        plt.xlim(0, 10)

        # Labels and title
        plt.xlabel("Age")
        plt.ylabel("R²")
        plt.title("R² vs Age of Stars")

        # Optional: Add grid for clarity
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.show()
    @staticmethod
    def temp(slope,intercept,x,y,name):
        plt.scatter(x, y, color="blue", label="Data points")

        # Regression line
        x_line = np.linspace(min(x), max(x), 100)  # smooth line
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")

        # Labels and legend
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Linear Regression Fit of {name}")
        plt.legend()
        plt.grid(True)
        plt.show()

Orbital_data = dataExtractor.dataplanet()
reg = regression(Orbital_data)
out = reg.dataHandler()
new = reg.recalc_with_correction()
Op = output()
#Op.data1(Orbital_data)
print("Calculated R^2 and mean values")
print("Name,No.ofPlanets,slope,intercept,R^2,MAE")
Op.data1(out)
print("-----------------------")
print("New Model Prediction before and after")
print("Name,Position of Predicted Planet,old R^2,new R^2,old MAE, new MAE, Timeperiod(days)")
Op.data1(new)
Op.data2(out)
Op.data3(Orbital_data,out)

