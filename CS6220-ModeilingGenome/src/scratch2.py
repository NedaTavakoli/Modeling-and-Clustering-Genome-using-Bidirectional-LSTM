import pandas as pd

# composite_df = pd.DataFrame(index=["tom", "jerry"])
composite_df = pd.DataFrame()

my_df = pd.DataFrame(index=["tom"])
my_df["c2"] = 2
my_df["c3"] = 4
# print(my_df)

total = my_df.values.sum()
# print(total)
final_df = my_df / total
# print(final_df)
composite_df = pd.concat([composite_df, final_df])

my_df = pd.DataFrame(index=["jerry"])
my_df["c2"] = 1
my_df["c4"] = 6
total = my_df.values.sum()
final_df = my_df / total

composite_df = pd.concat([composite_df, final_df])
print(composite_df)
