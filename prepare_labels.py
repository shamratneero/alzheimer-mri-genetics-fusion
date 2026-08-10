import pandas as pd

train_df = pd.read_csv('oasis_train_patients_metadata.csv')
test_df = pd.read_csv('oasis_test_patients_metadata.csv')

def to_binary(cls):
    return 'NonDemented' if cls == 'NonDemented' else 'Demented'

train_df['binary_label'] = train_df['class'].apply(to_binary)
test_df['binary_label'] = test_df['class'].apply(to_binary)

print("Train binary distribution:")
print(train_df['binary_label'].value_counts())
print("\nTest binary distribution:")
print(test_df['binary_label'].value_counts())

train_df.to_csv('train_binary.csv', index=False)
test_df.to_csv('test_binary.csv', index=False)