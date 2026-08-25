# 📥 Download Full Datasets

The four full benchmarks are **too large to ship with the repository**
(~2.6 GB in total). Please download the complete dataset archive from
Google Drive:

🔗 **https://drive.google.com/file/d/1TmtmuoAw9BzzBxKtsC0HCQRNRuS0PLrz/view?usp=sharing**

---

## 📦 What's Inside

The archive contains the **13 JSON files** that belong into `MMR/data/`:

| File | Size | Contents |
|------|------|----------|
| `longmemeval_s_all.json` | ~244 MB | LongMemEval-S (500 conversations) |
| `longmemeval_m_part1_all.json` … `longmemeval_m_part10_all.json` | ~240 MB each | LongMemEval-M, split into 10 shards |
| `locomo10_all_new.json` | ~2.0 MB | LoCoMo (10 conversations) |
| `LongMTBench_PLUS.json` | ~2.2 MB | Long-MT-Bench+ (11 conversations) |

---

## 🛠️ Setup

```bash
cd MMR/data
# 1) download the archive from the link above
# 2) unpack it into this folder
unzip <archive.zip>          # or: tar -xzf <archive.tar.gz>
```

After unpacking, `MMR/data/` should contain the 13 JSON files listed above.
The path is used by `run_pipeline.sh` as `<data_path>`, e.g.
`data/longmemeval_m_part1_all.json`.

---

## 🚀 Quick Start (no download needed)

For a quick test without downloading anything, use the small per-dataset
examples shipped in [`MMR/data_example/`](../data_example/):

```bash
cd MMR
bash run_pipeline.sh longmemeval_s data_example/LongMemEval_S_example.json checkpoints/bge-m3
```

Once the full datasets are in place, process LongMemEval-M shard by shard
with a distinct output tag so artifacts never overwrite each other:

```bash
cd MMR
for i in $(seq 1 10); do
    bash run_pipeline.sh longmemeval_m data/longmemeval_m_part${i}_all.json \
        checkpoints/bge-m3 part${i}
done
```

Results land in `outputs/part{1..10}/` for each shard.

---

> 💡 **Tip**: If Google Drive is not accessible, you can also fetch the
> benchmarks from their official sources and convert them with
> `MMR/data_utils.py`'s `load_dataset`.
