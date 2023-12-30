from datasets import load_dataset
dataset_name = "stas/c4-en-10k"
name = dataset_name.split('/')[-1]
ds = load_dataset(dataset_name, split='train')
ds.to_json(f"{name}.jsonl", orient="records", lines=True)