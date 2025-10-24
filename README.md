# ALL_BEAUTY

## Exporting report figures without binary uploads

The data-cleaning pipeline stores its tabular metrics in `reports/*.csv`. If you
need visualisations for documentation or a presentation, run:

```bash
python scripts/render_quality_figures.py --format svg
```

This renders lightweight SVG charts into `reports/figures/` that can be tracked
in Git because they are text files rather than binary PNGs. When a downstream
system requires PNG assets, generate them locally and emit base64-encoded
companions that can be downloaded safely:

```bash
python scripts/render_quality_figures.py --format png --export-base64
```

The command above writes the PNG files to `reports/figures/` (ignored by Git)
and stores textual `.b64` copies in `reports/figures/base64/`. To recover the
PNG on another machine, decode the base64 file, for example:

```bash
base64 -d reports/figures/base64/missing_rate_heatmap.b64 \
  > missing_rate_heatmap.png
```

This workflow avoids the “binary files are not supported” error during pull
request creation while still giving you a reproducible way to obtain the
figures.