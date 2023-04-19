# TransPPI

Protein-Protein Interaction prediction using an attention-based mechanism.

## How to use

This project provide you a model called trainsppi which defined a module called `PPITransformer`, which encode a protein using `ProteinEncoder`, which uses `GraphTransformerEncoder`.

### Preparing Data

You should prepare your ppi data in this way:

```txt
Q96NS5	P21854
Q6NT32	Q0P641
...
```

and store the positive and negative examples as `neg.txt` and `pos.txt` seperately in a single directory (you can name it whatever you want).

The second dataset you need to prepare is the pdb data for each protein in your ppi dataset. You need to store them in one single directory.

After that, you can you the pdb data to generate the prottrans features and coordinates data using `pdb_to_prottrans.py` and `pdb_to_coord.py` respectively. Don't forget to normalize the prottrans feature using `normalize_prottrans.py`

### Train the model

If you prepare and name your file in the same way as me, you can train the model directly using the script I provide:

```bash
python src/train.py
```

### Generate Results

Use the script `src/analyze_results`.
