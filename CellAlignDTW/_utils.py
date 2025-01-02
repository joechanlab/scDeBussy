import pandas as pd

def split_by_cutpoints(df, cutpoints, score_col):
    segments = [[] for _ in range(len(cutpoints) + 1)]
    
    for _, row in df.iterrows():
        value = row[score_col]
        for i, cutoff in enumerate(cutpoints):
            if value < cutoff:
                segments[i].append(row)
                break
        else:
            segments[-1].append(row)
    
    segments = [pd.DataFrame(segment) for segment in segments]
    return segments