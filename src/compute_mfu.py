import wandb

gpu_constant = 154.85
parameters = 125000000 / 10**9
layers = 12
heads = 12
head_dimension = 64 / 10**6

api = wandb.Api()
runs = api.runs(path="nlp_benchmarks/fast_deep_learning", filters={"tags": {"$in": ["report"]}})

for run in runs:
    sequence_length = run.config["max_sequence_length"] / 10**3
    R = gpu_constant / (6*parameters + 12*layers*heads*head_dimension*sequence_length)
    mean_tokens = run.summary["TokensPerSecond"]["mean"] / 1000
    mfu = mean_tokens / R 
    print(f"{run.name} has MFU {mfu:.2f}")
    run.summary["mean_mfu"] = mfu
    run.summary["max_tokens_theoretical"] = R*1000
    run.summary.update()
    