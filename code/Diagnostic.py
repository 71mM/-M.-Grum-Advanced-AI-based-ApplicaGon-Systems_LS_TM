import pickle
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

with open("../images/knowledgeBase/currentOlsSolution.pkl", "rb") as file:
    ols_model= pickle.load(file)

cls = LinearRegDiagnostic(ols_model)
cls(plot_context="seaborn-v0_8-paper")
vif_table = cls.vif_table()
with PdfPages('UE_04_App2_DiagnosticPlots.pdf') as pdf:

    vif_table_plot = cls(plot_context="seaborn-v0_8-paper")[1]


    pdf.savefig(vif_table_plot)
    plt.close(vif_table_plot)


    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('tight')
    ax.axis('off')


    table = ax.table(cellText=vif_table.values,
                     colLabels=vif_table.columns,
                     loc='center', cellLoc='center', colColours=['#f1f1f1'] * len(vif_table.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(vif_table.columns))))


    pdf.savefig(fig)
    plt.close(fig)
