.PHONY: data clean

## Recreate processed, augmented datasets and quality reports
data:
	python -m scripts.clean_data

clean:
	rm -f data/processed/*.csv data/processed/*.jsonl \
	   data/augmented/*.jsonl reports/*.csv
