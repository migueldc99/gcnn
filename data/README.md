# Data Directory

This directory holds the molecular dataset (QM9 / GDB-9).

## How to obtain the dataset

1. Download `dsgdb9nsd.xyz.tar` from:
   - https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

2. Place the file at:
   ```
   data/original_dataset/dsgdb9nsd.xyz.tar
   ```

3. Run `notebooks/1_gen_database.ipynb` to extract and split the data into train/validation/test sets.

## Structure after running notebook 1

```
data/
├── original_dataset/
│   └── dsgdb9nsd.xyz.tar
├── dataset/                  # All extracted .xyz files
├── training_set/             # ~81% of data
├── validation_set/           # ~9% of data
└── test_set/                 # ~10% of data
```
