import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── linhas do diagrama ────────────────────────────────────────────────────────
linhas = [
    # (prefixo_arvore, nome_arquivo, descricao)
    ('',   'projeto_mer/',    ''),
    ('├─ ', 'dataset.py',     'extração espectral, normalização, augmentação, Dataset PyTorch'),
    ('├─ ', 'model.py',       'ResNet18MER e função build_model'),
    ('├─ ', 'preprocess.py',  'pré-computa e salva os espectrogramas em disco (executado uma vez)'),
    ('├─ ', 'train.py',       'pipeline completo: 5-fold CV, balanceamento, loss, otimização'),
    ('├─ ', 'predict.py',     'inferência em músicas novas com os modelos treinados'),
    ('├─ ', 'benchmark.py',   'coleta de métricas e serialização dos resultados'),
    ('└─ ', 'benchmark.ipynb','comparação visual dos três modelos (gráficos do Capítulo 5)'),
]

# ── cores ─────────────────────────────────────────────────────────────────────
BG        = '#1E1E2E'   # fundo escuro
TREE_CLR  = '#6C7086'   # ramos da árvore
ROOT_CLR  = '#CDD6F4'   # nome do projeto
FILE_CLR  = '#89B4FA'   # nomes de arquivo .py
NB_CLR    = '#F38BA8'   # notebook .ipynb
ARROW_CLR = '#45475A'   # seta ←
DESC_CLR  = '#A6ADC8'   # descrição

# ── figura ────────────────────────────────────────────────────────────────────
n = len(linhas)
fig_h = 0.42 * n + 0.5
fig, ax = plt.subplots(figsize=(9.5, fig_h))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

PAD   = 0.28          # margem lateral em dados de ax
LINHA = 1.0 / n       # altura de cada linha
TOP   = 1.0 - 0.5/n  # y da primeira linha (centrado)

for i, (prefixo, nome, desc) in enumerate(linhas):
    y = TOP - i * LINHA

    if i == 0:
        # raiz do projeto
        ax.text(PAD, y, nome, color=ROOT_CLR,
                fontsize=11, fontweight='bold', fontfamily='monospace',
                va='center', transform=ax.transAxes)
        continue

    eh_nb   = nome.endswith('.ipynb')
    eh_last = (i == len(linhas) - 1)
    f_cor   = NB_CLR if eh_nb else FILE_CLR

    # ramo da árvore
    ax.text(PAD, y, prefixo, color=TREE_CLR,
            fontsize=10, fontfamily='monospace',
            va='center', transform=ax.transAxes)

    # nome do arquivo
    ax.text(PAD + 0.038, y, nome, color=f_cor,
            fontsize=10, fontweight='bold', fontfamily='monospace',
            va='center', transform=ax.transAxes)

    if desc:
        # seta e descrição
        ax.text(PAD + 0.038 + len(nome)*0.018 + 0.012, y,
                '← ', color=ARROW_CLR,
                fontsize=10, fontfamily='monospace',
                va='center', transform=ax.transAxes)
        ax.text(PAD + 0.038 + len(nome)*0.018 + 0.033, y,
                desc, color=DESC_CLR,
                fontsize=9, fontfamily='monospace',
                va='center', transform=ax.transAxes)

# borda sutil ao redor da figura
for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_visible(False)
fig.patch.set_linewidth(0)

os.makedirs('figuras_tcc', exist_ok=True)
out = 'figuras_tcc/estrutura_codigo.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Salvo em: {out}')
