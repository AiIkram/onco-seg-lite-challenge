# Public data

Organisers generate the contents of this folder with `scripts/prepare_data.py` (see `DESIGNER_MANUAL.md` §3.3):

```
train/images/<id>.png     256x256 grayscale ultrasound
train/masks/<id>.png      0 = background, 255 = lesion (multi-lesion masks merged)
train_metadata.csv        image_id, diagnosis (benign | malignant | normal)
test/images/<id>.png      256x256 grayscale ultrasound
test_ids.csv              image_id
sample_submission.csv     image_id, rle  (all empty)
```

Source: BUSI. Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *Dataset of breast ultrasound images.*
Data in Brief 28 (2020) 104863. doi:10.1016/j.dib.2019.104863
