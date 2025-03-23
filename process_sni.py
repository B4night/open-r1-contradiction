from datasets import load_dataset, concatenate_datasets

ds = load_dataset("stanfordnlp/snli")

new_ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])

print(new_ds)

# if new_ds['label'] == 0 or new_ds['label'] == 1, make them 0. If new_ds['label'] == 2, make it 1
new_ds = new_ds.map(
    lambda example: {'label': 0 if example['label'] in [0, 1] else 1})

# new_ds has 3 columns: 'label', 'premise', 'hypothesis'
# Generate a new column text and remove premise and hypothesis. text = premise + ' ' + hypothesis

new_ds = new_ds.map(
    lambda example: {'text': example['premise'] + ' ' + example['hypothesis']})

# remove premise and hypothesis
new_ds = new_ds.remove_columns(['premise', 'hypothesis'])

# select the longest 5000 examples, their text is the longest
# new_ds = new_ds.sort('text', reverse=True)
new_ds = new_ds.select(range(5000))

print(new_ds)

# split the dataset into train and test
new_ds = new_ds.train_test_split(test_size=0.2, shuffle=True)

print(new_ds)


# convert 'label' to 'completion'
# new_ds = new_ds.rename_column('label', 'completion')

# Add a new column, "completion", all values are empty
# new_ds = new_ds.map(lambda example: {**example, 'completion': ''})

# make 'text' the first column
new_ds = new_ds.rename_column('text', 'messages')
new_ds = new_ds.rename_column('label', 'classification')

print(new_ds)

new_ds.save_to_disk('./data/snli_sampled')
