"""Matplotlib settings shared by every figure of Section 7 and Appendix C.

The figures are drawn 119mm wide, the figure width of the journal, and included
at 1:1, so the font sizes set here are the sizes the reader sees; the journal
asks for lettering of 8 to 12pt.  Helvetica is the intended face, and the list
below falls back to faces with its metrics.  ``ensure_helvetica`` reports which
face a given system picks.
"""
import matplotlib as mpl

def set_paper_style():
    mpl.rcParams.update({
        # ---- Font (Helvetica) ----
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "TeX Gyre Heros",
                            "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",   # sans-serif maths, to match Helvetica

        # ---- Figure / save ----
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # ---- Sizes (paper-friendly) ----
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,

        # ---- Lines / markers ----
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "text.color": "black",

        # ---- Ticks ----
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,


        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "axes.labelpad": 1.0,
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,

        # ---- Grid ----
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # ---- PDF/PS text handling: keep the text vectorial ----
        "pdf.fonttype": 42,   # embed as TrueType
        "ps.fonttype": 42,
    })

def save_pdf(fig, path):
    """Save with the bbox, padding and dpi fixed by the rcParams above."""
    fig.savefig(path)


def ensure_helvetica(prefer_clone: bool = True) -> str:
    """
    Returns the first sans-serif face of the Helvetica stack that is available
    on this system.  Nimbus Sans and TeX Gyre Heros have the metrics of Helvetica.
    """
    from matplotlib import font_manager
    candidates = (["Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Arial", "DejaVu Sans"]
                  if prefer_clone else
                  ["Helvetica", "Arial", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans"])
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"
