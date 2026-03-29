import pandas as pd

def generate_csv_file(path="tasks.csv"):
    data = [
        ["Fix login bug on website", "high"],
        ["Update user profile page", "low"],
        ["Implement new API endpoint", "high"],
        ["Refactor old code", "low"],
    ]

    df = pd.DataFrame(data, columns=["task_description", "priority"])
    df.to_csv(path, index=False)
    return path