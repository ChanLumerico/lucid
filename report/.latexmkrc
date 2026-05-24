# latexmk configuration for Lucid tech report
# Engine: XeLaTeX + biber

$pdf_mode  = 5;   # 5 = xelatex
$xelatex   = 'xelatex -shell-escape -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

# Use biber for biblatex
$biber       = 'biber %O %S';
$bibtex_use  = 2;

# Aux files to clean
$clean_ext = 'aux bbl bcf blg fdb_latexmk fls log out run.xml synctex.gz toc lof lot idx ilg ind acn acr alg glg glo gls ist xdy';

# Continuous-build preview command (no auto-open; user picks viewer)
$preview_continuous_mode = 1;

# Don't open viewer automatically
$pdf_previewer = '';
