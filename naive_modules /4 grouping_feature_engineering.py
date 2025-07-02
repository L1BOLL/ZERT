import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Grouping by and averaging expression of each transcript

ddd = filtered_df.groupby("type")
means = ddd.mean()
means

# Getting differential expression profile of transcripts between types and controls

la = means.diff().iloc[-1].to_list()
# Upregulated in types
urc = []
# Downregulated in types
drc = []
for i in (la):
    if i>0:
        urc.append(i)
    else:
        drc.append(-i)
urc.sort(reverse = True)
drc.sort(reverse = True)  

len(la)

# tdf dataframe is filtered_df ordered by differential expression, from highest to lowest
tdf = filtered_df.transpose()
tdf.drop("type", axis = 0, inplace = True)
tdf['regulation'] = la
tdf.sort_values('regulation', key = abs, ascending = False, inplace = True)
tdf

# gdf has only the 20 transcripts with the highes differential expression
gdf = tdf[:20]
# bmg = BioMarker Group
bmg = gdf.index.to_list()
gdf

bmg

filtered_df[bmg]
