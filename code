# Import Data

# import kaggle
# import zipfile

# kaggle datasets download -d adilshamim8/cost-of-international-education
# with zipfile.ZipFile("cost-of-international-education.zip", "r") as zip_ref:
#     zip_ref.extractall("cost-of-international-education")

# Create EDAs

import pandas as pd                 # Data manipulation
import numpy as np                  # Number operations
import matplotlib.pyplot as plt     # Plotting
import seaborn as sns               # Data visualization
from scipy.stats import linregress  # Linear regression
import geopandas as gpd             # Map data

df = pd.read_csv("cost-of-international-education/International_Education_Costs.csv")
cost_cols = ['Tuition_USD','Living_Cost_Index','Rent_USD','Visa_Fee_USD','Insurance_USD']
df = df.dropna(subset=cost_cols)
df = df[(df[cost_cols] != 0).all(axis=1)]

backgroundColor = "#F2E9E4"
blue, red, gold = "#1f77b4", "#d62728", "#ff9f1c"
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

plt.style.use("ggplot")

def setdefaultcolor(xsize = None, ysize = None):
    if (xsize is not None):
        plt.figure(facecolor=backgroundColor, figsize=(xsize, ysize))
    else:
        plt.figure(facecolor=backgroundColor)

# I. Tuition Fees Distribution (Tuition Fee to Frequency) (Histogram)
setdefaultcolor()

median = df["Tuition_USD"].median()
mean = df["Tuition_USD"].mean()

sns.histplot(df["Tuition_USD"], bins=20, kde=True, color = blue)
plt.title("Tuition Fees Distribution (USD)", weight = "bold", fontsize = 14)
plt.xlabel("Tuition (USD)", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.axvline(median, label = f"Median: {median:.0f} USD", linestyle = "--", linewidth = 1.5, color = red)
plt.axvline(mean, label = f"Mean: {mean:.0f} USD", linestyle = "-", linewidth = 1.5, color = gold)
plt.legend()
plt.tight_layout()
plt.show()

# II. Average Tuition Fees per Country (Country to Tuition Fee) (Bar)

avg_tuition_country = df.groupby("Country")["Tuition_USD"].mean().sort_values(ascending=True)

fig, ax = plt.subplots(1, 1, figsize = (30, 20), facecolor = backgroundColor)
avg_tuition_country.plot(kind = "barh", alpha = 0.7, zorder = 1, color = blue)
ax.set_title("Average Tuition Fees per Country (USD)", weight = "bold", fontsize = 14)
ax.set_xlabel("Tuition (USD)")
ax.set_ylabel("Country")
plt.yticks(rotation=45)
for i in range(5):
    ax.axvline(10000 * i, linestyle="--", linewidth = 1.5, color = red, zorder = 0)
plt.tight_layout()
plt.show()

# III. Tuition Fees Distribution by Program (Program to Tuition Fee) (Boxplot)
setdefaultcolor(20, 10)

median_order = df.groupby("Program")["Tuition_USD"].median().sort_values(ascending=True)

sns.boxplot(
    x = "Program",
    y = "Tuition_USD",
    data=df,
    order = median_order.index)
plt.title("Tuition Fees Distribution by Program (USD)", weight = "bold")
plt.xlabel("Program")
plt.ylabel("Tuition (USD)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# IV. Living Cost vs Tuition (Living Cost to Tuition Fee) (Scatter + Line)
setdefaultcolor()

x = df["Living_Cost_Index"]
y = df["Tuition_USD"]

slope, intercept, r, p, stderr = linregress(x, y)

plt.scatter(x, y, marker = ".", alpha = 0.7)
plt.plot(
    x,
    intercept + slope * x,
    label = f"y = {intercept:.0f} + {slope:.0f}x\nR²={r**2:.2f}",
    color = blue,
    linestyle = "--"
)
plt.title("Living Cost vs Tuition", weight = "bold")
plt.xlabel("Living Cost Index")
plt.ylabel("Tuition (USD)")
plt.legend()
plt.show()

# V. Correlation Between Cost Components (Tuition Fee, Living Cost, Rent, Visa Fee, Insurance) (Heatmap)
setdefaultcolor(12, 9)

corr_components = df[[
    "Tuition_USD",
    "Living_Cost_Index",
    "Rent_USD",
    "Visa_Fee_USD",
    "Insurance_USD"]].corr()

sns.heatmap(corr_components, annot=True, fmt=".2f", cmap="vlag", annot_kws={"size": 20})
plt.title("Correlation Between Cost Components", weight = "bold", fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

# VI. Global Map of Annual Study Abroad Cost (Map to Tuition Fee) (Map)
def estimate_annual_cost(row):
    tuition = row['Tuition_USD'] / row['Duration_Years']
    living = row['Living_Cost_Index'] / 100 * 12000  # baseline $12k/year
    rent = row['Rent_USD'] * 12
    visa = row['Visa_Fee_USD']
    insurance = row['Insurance_USD']
    return tuition + living + rent + visa + insurance

df["Estimated_Annual_Cost"] = df.apply(estimate_annual_cost, axis = 1)

world = gpd.read_file("https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson")
country_cost = df.groupby("Country")["Estimated_Annual_Cost"].mean().reset_index()
merged = world.merge(country_cost, left_on="name", right_on="Country", how="left")

available_data = merged[merged["Estimated_Annual_Cost"].notna()]
missing_data = merged[merged["Estimated_Annual_Cost"].isna()]

fig, ax = plt.subplots(1, 1, figsize = (14, 8), facecolor = backgroundColor)

missing_data.plot(
    ax=ax,
    color="lightgrey",
)

available_data.plot(
    column="Estimated_Annual_Cost",
    ax=ax,
    legend=True,
    cmap="viridis",
    linewidth=0.5,
    edgecolor="black",
)

ax.set_title("Global Map of Annual Study Abroad Cost", weight = "bold", fontsize = 24)
plt.tight_layout()
ax.axis("off")
plt.show()
