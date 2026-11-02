"""Builds the IRIS (Springer LNCS) version of the paper from the TMLR draft, so both say the same.

    python make_lncs.py        # reads ../paper_tmlr/main.tex and refs.bib, writes main.tex and refs.bib here

Differences from the TMLR version: LNCS class, author block and keywords; \\cite instead of
\\citep; the named code link instead of the anonymous one; no broader-impact section; wide
tables shrink to the narrower LNCS text width.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "paper_tmlr")

PREAMBLE = r"""% IRIS'27 version, generated from ../paper_tmlr/main.tex by make_lncs.py; edit the TMLR source
% and re-run rather than editing this file. Springer LNCS class until the IRIS template arrives.
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage[hidelinks]{hyperref}

\newcommand{\focus}{\textsc{focus}}
\newcommand{\fixed}{\textsc{fixed}}
\newcommand{\densewin}{\textsc{dense-win}}

\begin{document}

\title{Anchors That Move Without Collapsing:\\Exclusive-Region Anchor Routing for Vision Transformers}
\titlerunning{Anchors That Move Without Collapsing}
\author{Aadi Gupta}
\authorrunning{A. Gupta}
\institute{Scaler School of Technology, Bengaluru, India\\
\email{aadi.24bcs10167@sst.scaler.com}}

\maketitle
"""

KEYWORDS = r"\keywords{Vision transformers \and Anchor tokens \and Token routing \and Representation collapse \and High-resolution images}"
CODE_LINK = r"https://github.com/ARRY7686/Partitioned-Residual-Anchor-Routing-Transformer"


def convert(tex):
    body = tex[tex.index(r"\begin{abstract}"):]
    body = body.replace(r"\end{abstract}", KEYWORDS + "\n" + r"\end{abstract}", 1)
    body = re.sub(r"\\citep\{", r"\\cite{", body)
    # Broader impact is a TMLR section; IRIS has none.
    body = re.sub(r"\\subsubsection\*\{Broader Impact Statement\}.*?(?=\\paragraph\{Code availability\.\})",
                  "", body, flags=re.S)
    body = re.sub(r"\\url\{https://anonymous\.4open\.science/r/XXXX\} \(anonymised for review\)\.[^\n]*",
                  r"\\url{" + CODE_LINK + "}.", body)
    body = body.replace(r"\bibliographystyle{tmlr}", r"\bibliographystyle{splncs04}")
    body = re.sub(r"\\begin\{tabular\}(\{[^}]*\})", r"\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}\1", body)
    body = body.replace(r"\end{tabular}", r"\end{tabular}" + "\n" + r"\end{adjustbox}")
    return PREAMBLE + "\n" + body


tex = open(os.path.join(SRC, "main.tex"), encoding="utf-8").read()
open(os.path.join(HERE, "main.tex"), "w", encoding="utf-8").write(convert(tex))
bib = open(os.path.join(SRC, "refs.bib"), encoding="utf-8").read()
open(os.path.join(HERE, "refs.bib"), "w", encoding="utf-8").write(bib)
print("wrote", os.path.join(HERE, "main.tex"))
