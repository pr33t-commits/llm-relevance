import numpy as np



def prob_diff(df, prefix):
    return df[f"{prefix}_correct_prob"] - df[f"{prefix}_wrong_prob"]

def logit_diff(df, prefix):
    return df[f"{prefix}_correct_logit"] - df[f"{prefix}_wrong_logit"]


def avg_prob_diff(df, prefix):
    return prob_diff(df, prefix).values.mean()

def avg_logit_diff(df, prefix):
    return logit_diff(df, prefix).values.mean()


def metric_prob_diff(df):
    return prob_diff(df, "patched") - prob_diff(df, "corrupted")

def metric_logit_diff(df):
    return logit_diff(df, "patched") - logit_diff(df, "corrupted")

def metric_prob_diff_normalized(df):
    avg_corrupted = avg_prob_diff(df, "corrupted")
    avg_clean = avg_prob_diff(df, "clean")
    return (prob_diff(df, "patched") - avg_corrupted) / (avg_clean - avg_corrupted)

def metric_prob_diff_normalized_2(df):
    return (prob_diff(df, "patched") - prob_diff(df, "corrupted")) / (prob_diff(df, "clean") - prob_diff(df, "corrupted"))

def metric_logit_diff_normalized(df):
    avg_corrupted = avg_logit_diff(df, "corrupted")
    avg_clean = avg_logit_diff(df, "clean")
    results = (logit_diff(df, "patched") - avg_corrupted) / (avg_clean - avg_corrupted)
    results[np.isnan(results)] = 0
    return results

def metric_logit_diff_normalized_2(df):
    results = (logit_diff(df, "patched") - logit_diff(df, "corrupted")) / (logit_diff(df, "clean") - logit_diff(df, "corrupted"))
    results[np.isnan(results)] = 0
    return results


def metric_prob_diff_pp(df):
    return (df[f"patched_correct_prob"] - df[f"clean_correct_prob"]) / df[f"clean_correct_prob"].values.mean()

def metric_logit_diff_pp(df):
    return (df[f"patched_correct_logit"] - df[f"clean_correct_logit"]) / df[f"clean_correct_logit"].values.mean()


def metric_logit_diff_normalized_pp(df):
    avg_clean = avg_logit_diff(df, "clean")
    avg_corrupted = avg_logit_diff(df, "corrupted")
    results = (logit_diff(df, "patched") - logit_diff(df, "clean")) / (avg_corrupted - avg_clean)
    results[np.isnan(results)] = 0
    return results