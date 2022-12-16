import pandas as pd

df = pd.read_csv("ent.petster-hamster", delim_whitespace=True, encoding="ansi")
df["dat.country"] = df["dat.hometown"].str.split(",").str[-1]
# Remove the ; for US and Canada
df["dat.country"] = df["dat.country"].str.split(";").str[-1].str.strip()
print("debug")