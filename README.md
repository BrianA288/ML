# Materials Volatility research page

A static project website built from the supplied `ZIPMLAssignment.zip` results. No build step is required.

## Publish with GitHub Pages

1. Create a public GitHub repository and upload the files in this directory to its root (keep `assets/` intact).
2. In **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/ (root)`, then save.
3. The site will appear at `https://YOUR-USERNAME.github.io/REPOSITORY-NAME/` after GitHub finishes publishing.

To preview locally, run `python -m http.server 8000` in this directory and visit `http://localhost:8000`. Opening `index.html` directly may prevent `metrics.json` from loading due to browser file restrictions.

The chart PNGs and metrics are derived from the uploaded project; the site does not include raw market data or notebooks. The original README in the ZIP contains notebook filenames that differ from the actual archive, so use the archive's notebook paths when linking the full project.
