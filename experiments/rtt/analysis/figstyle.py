# Shared figure style for the paper, modelled on common COLING/ACL figures: Times-like serif text, black frames,
# Office-style pastel fills with dashed outlines, dash-dot stage containers, hollow block arrows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, FancyArrow, Circle, Rectangle

SERIF = 'Liberation Serif'          # metric-compatible with Times New Roman
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [SERIF, 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.grid': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

INK = '#000000'
GREY_TXT = '#404040'
# pastel fills (Office theme tints) and matching darker accents
FILL = {'yellow': '#fff2cc', 'blue': '#dae3f3', 'purple': '#e4d7ef', 'orange': '#fbe5d6', 'grey': '#e7e6e6',
        'green': '#e2f0d9', 'lavender': '#dcd7f2'}
ACC = {'yellow': '#bf9000', 'blue': '#2f5597', 'purple': '#7030a0', 'orange': '#c55a11', 'grey': '#7f7f7f',
       'green': '#548235', 'lavender': '#4a3aa7'}
# roles (kept consistent with the case-study frames)
ROLE = {'A': ('orange', '#eb6834'), 'L': ('blue', '#2a78d6'), 'O': ('grey', '#8a8984')}
BAR = {'main': '#4472c4', 'light': '#8faadc', 'grey': '#a5a5a5', 'orange': '#ed7d31'}


def box(ax, x, y, w, h, text='', fill='white', ec='#404040', ls=(0, (3, 2)), lw=0.8, fs=8, r=0.8, color=INK,
        z=2, **kw):
    """Rounded box with (by default) a dashed outline, centred text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fill, ec=ec, lw=lw,
                                ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, zorder=z + 1,
                linespacing=1.1, **kw)


def token(ax, x, y, w, h, label, fill, fs=8, z=3):
    """Feature/token square: pastel fill, solid dark outline, italic math label."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec='#1f1f1f', lw=0.8, zorder=z))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs, zorder=z + 1)


def container(ax, x, y, w, h, label=None, label_y=None, fs=8.5, r=2.5):
    """Dash-dot rounded stage container with its name centred below."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc='none', ec=INK,
                                lw=0.8, ls=(0, (5, 1.5, 1, 1.5)), zorder=1))
    if label:
        ax.text(x + w / 2, (y - 2.2) if label_y is None else label_y, label, ha='center', va='top', fontsize=fs,
                linespacing=1.1)


def arrow(ax, x1, y1, x2, y2, lw=0.7, ms=7, color=INK, z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def line(ax, xs, ys, lw=0.7, color=INK, z=4, **kw):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle='butt', **kw)


def hollow_arrow(ax, x, y, dx, dy, width, head_width, head_length, lw=0.8, z=3):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=width, head_width=head_width, head_length=head_length,
                            length_includes_head=True, fc='white', ec=INK, lw=lw, zorder=z))


def frame_axes(ax):
    """Plain black box frame, no grid (chart style)."""
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(INK); s.set_linewidth(0.6)
    ax.grid(False)
