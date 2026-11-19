# Project report

Write the report directly in `main.tex`. It uses the supplied ACM template
with the two-column `sigconf` layout and no ACM publication metadata.

Build from the repository root:

```sh
make -C report
```

This produces `report/main.pdf`. The local TeX installation is missing
required packages, starting with `xkeyval.sty`, so PDF compilation has not
been verified. Alternatively, upload this folder to Overleaf and select
`main.tex` with pdfLaTeX.

Put figures in `figures/` and bibliography entries in `references.bib`.
For example, cite an entry named `dataset-paper` with `\cite{dataset-paper}`.

The template files are unchanged copies from `acmart-primary.zip`.
See `TEMPLATE-README` for upstream instructions and license information.
