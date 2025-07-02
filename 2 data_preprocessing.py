# Exclude Sample_ID and last type
transcript_cols = df.columns[1:-1]

# Count zeros per transcript
zero_counts = (df[transcript_cols] == 0).sum(axis=0)

# Select 1000 transcripts with the fewest zeros
#top_transcripts = zero_counts.nsmallest(int(df.shape[1]/2)).index

# To ismplify the computation : 
top_transcripts = zero_counts.nsmallest(1000).index

# Build final column list: first column + selected transcripts + last column
final_cols = [df.columns[0]] + list(top_transcripts) + [df.columns[-1]]

# Subset DataFrame
df_subset = df[final_cols]

df = df_subset
df.shape

# split data to unbias

from sklearn.model_selection import train_test_split

train_val, test_set = train_test_split(
    df,
    test_size=0.2,
    stratify=df["type"],  # to maintain class balance
    random_state=42
)

df = train_val
df.shape