"""Portfolio fan-card cover: what the IKEA model promised vs what it earned."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))["ikea"]
rows = [("PROMISED", d["promised"], "#ffffff", 0.32), ("EARNED", d["policies"][1]["value"], "#9D9FFF", 1.0),
        ("RANDOM", d["policies"][0]["value"], "#ffffff", 0.32)]
W, H = 1536, 928
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); fig.patch.set_facecolor("#0b0b0d")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#0b0b0d"); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 60)
top = rows[0][1]
for i, (label, v, color, a) in enumerate(rows):
    y = 50 - i * 10.5
    ax.text(7, y + 3.4, label, color="white", alpha=0.5, fontsize=17, family="monospace")
    ax.add_patch(plt.Rectangle((7, y - 1.6), 78, 3.2, color="white", alpha=0.06, lw=0))
    ax.add_patch(plt.Rectangle((7, y - 1.6), 78 * v / top, 3.2, color=color, alpha=a, lw=0))
    ax.text(88, y - 1.2, f"${v:.2f}", color=color if label == "EARNED" else "white", alpha=1 if label == "EARNED" else 0.8,
            fontsize=24, family="monospace")
fig.savefig(sys.argv[2], facecolor=fig.get_facecolor())
