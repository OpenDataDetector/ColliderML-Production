# Vendored fonts

`DejaVuSans-Bold.ttf`, `DejaVuSans-BoldOblique.ttf` — the bold faces of DejaVu
Sans, vendored so the ODD experiment label (`odd_plot_style.odd_label`) always
renders bold/bold-italic.

**Why vendored:** some stripped matplotlib installs (e.g. the NERSC `hep4m2`
conda env) ship only the *regular* DejaVu Sans face. When the bold face is
missing, matplotlib silently falls back to regular, so `fontweight="bold"` (and
atlasify's bold-italic badge) render un-bold with no error. `ensure_bold_font()`
in `odd_plot_style.py` registers these bundled faces when called (it is invoked by
`paper_style()`) if the env lacks a bold DejaVu face, making the label robust in
any environment.

DejaVu Fonts are freely redistributable (Bitstream Vera / public-domain-derived
license). See https://dejavu-fonts.github.io/License.html
