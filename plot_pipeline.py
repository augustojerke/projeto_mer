import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG    = "#0F1117"
CARD  = "#1C1F2E"
WHITE = "#F0F2FF"
DIM   = "#5A5E78"

steps = [
    ("#4A90D9", "1. Divisão dos Dados"),
    ("#9B59B6", "2. Validação Cruzada  (KFold=5)"),
    ("#E8A838", "3. Balanceamento de Classes"),
    ("#52B788", "4. Aumento de Dados"),
    ("#E07B54", "5. Função de Perda"),
    ("#3498DB", "6. Otimizador  (AdamW)"),
    ("#1ABC9C", "7. Agendamento do LR"),
    ("#E74C3C", "8. Execução  (BF16 + Clip)"),
    ("#F39C12", "9. Early Stopping"),
    ("#2ECC71", "10. Seleção do Modelo Final"),
]

COLS  = 3
CW    = 3.3    # card width
CH    = 0.52   # card height
HG    = 0.55   # horizontal gap
VG    = 0.90   # vertical gap between rows
MX    = 0.55   # left/right margin
MY    = 0.55   # top margin below title

rows  = [steps[i:i+COLS] for i in range(0, len(steps), COLS)]
NROWS = len(rows)

total_w = COLS * CW + (COLS - 1) * HG + 2 * MX
total_h = MY + 0.55 + NROWS * CH + (NROWS - 1) * VG + 0.4

fig = plt.figure(figsize=(total_w, total_h), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h)
ax.axis("off")

ax.text(total_w / 2, total_h - 0.28,
        "Protocolo de Treinamento",
        fontsize=12, fontweight="bold", color=WHITE,
        ha="center", va="center", fontfamily="monospace")

def card_xy(row_idx, col_idx, n_in_row):
    row_width = n_in_row * CW + (n_in_row - 1) * HG
    offset    = (total_w - row_width) / 2  
    x = offset + col_idx * (CW + HG)
    y = total_h - MY - 0.55 - row_idx * (CH + VG) - CH
    return x, y

positions = []
for ri, row in enumerate(rows):
    for ci, (color, label) in enumerate(row):
        x, y = card_xy(ri, ci, len(row))
        positions.append((x, y, color, label))

        box = FancyBboxPatch((x, y), CW, CH,
                             boxstyle="round,pad=0,rounding_size=0.08",
                             linewidth=1.3, edgecolor=color,
                             facecolor=CARD, zorder=3)
        ax.add_patch(box)

        bar = FancyBboxPatch((x, y), 0.11, CH,
                             boxstyle="round,pad=0,rounding_size=0.06",
                             linewidth=0, facecolor=color, alpha=0.9, zorder=4)
        ax.add_patch(bar)

        ax.text(x + 0.22, y + CH / 2, label,
                fontsize=11, color=WHITE, va="center",
                zorder=5, fontfamily="monospace")

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=DIM,
                                lw=1.1, mutation_scale=9), zorder=6)

for i in range(len(positions) - 1):
    x1, y1, *_ = positions[i]
    x2, y2, *_ = positions[i + 1]

    ri_cur  = i // COLS
    ri_next = (i + 1) // COLS

    if ri_cur == ri_next:
        draw_arrow(ax, x1 + CW, y1 + CH / 2, x2, y2 + CH / 2)
    else:
        bx = x1 + CW / 2   
        by = y1            
        tx = x2 + CW / 2  
        ty = y2 + CH   
        mid_y = (by + ty) / 2

        ax.plot([bx, bx], [by, mid_y],  color=DIM, lw=1.1, zorder=6)
        ax.plot([bx, tx], [mid_y, mid_y], color=DIM, lw=1.1, zorder=6)
        draw_arrow(ax, tx, mid_y, tx, ty)

fig.savefig("figuras_tcc/pipeline_treinamento.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
print("Salvo: figuras_tcc/pipeline_treinamento.png")
