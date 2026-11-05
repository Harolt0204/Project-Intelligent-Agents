"""Resumen de incertidumbre desde medias ya guardadas; no carga agentes ni entrena."""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.intervalos import ci_record

ROOT=Path(__file__).resolve().parents[1]


def main():
    pointer=json.loads((ROOT/'results/tier1_actual.json').read_text(encoding='utf-8'))
    tier1=json.loads((ROOT/pointer['comparison']/'summary.json').read_text(encoding='utf-8'))
    lunar=json.loads((ROOT/'results/tier23_mejora_v1/summary.json').read_text(encoding='utf-8'))
    rows=[]
    for row in tier1:
        values=[row['final_returns_by_seed'][str(seed)] for seed in (42,43,44)]
        rows.append({'tier':1,'method':row['method'],'sample':'final','values':values,**ci_record(values)})
    for row in lunar['groups']:
        for sample,key in [('test','test_mean'),('confirmation','confirmation_mean')]:
            values=[r[key] for r in row['runs']]
            rows.append({'tier':2 if row['group']=='base' else 3,'method':'DQN LunarLander' if row['group']=='base' else 'Double DQN LunarLander',
                         'sample':sample,'values':values,**ci_record(values)})
    paired=[]
    for key in ('test_mean','confirmation_mean'):
        values=[b[key]-a[key] for a,b in zip(lunar['groups'][0]['runs'],lunar['groups'][1]['runs'])]
        paired.append({'comparison':'Double minus DQN','sample':key,'values':values,**ci_record(values)})
    folder=ROOT/'results/analisis_intervalos'
    folder.mkdir(exist_ok=True)
    result={'method':'95% pointwise Student t, n=3 training seeds, df=2', 'runs_retrained':0,
        'rows':rows,'paired_differences':paired}
    (folder/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,tier,title,threshold in [(axes[0],1,'CartPole: modelos finales',475),(axes[1],2,'LunarLander: prueba nueva',200)]:
        selected=[r for r in rows if (r['tier']==1 if tier==1 else r['tier'] in (2,3)) and r['sample'] in ('final','test')]
        labels=[r['method'].replace('Q-learning ','Q ').replace(' intervalos','').replace(' LunarLander','') for r in selected]
        for i,r in enumerate(selected):
            ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['ci95_low']],[r['ci95_high']-r['mean']]],fmt='o',capsize=4,color='#2563eb')
        ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis()
        ax.axvline(threshold,ls='--',color='#64748b',label=f'Objetivo {threshold}')
        ax.set_xlabel('Retorno medio de tres semillas')
        ax.set_title(title)
        ax.grid(axis='x',alpha=.2)
        ax.legend(loc='best',fontsize=8)
    fig.suptitle('Medias e IC 95% orientativos (t, n=3; 100 episodios por modelo)')
    for ext in ('png','pdf'): fig.savefig(folder/f'resultados_ic95.{ext}',dpi=160)
    plt.close(fig)
    lines=['# Intervalos calculados desde resultados guardados','',
        '| Tier | Método | Muestra | Media | IC 95% |','|---|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['tier']} | {r['method']} | {r['sample']} | {r['mean']:.2f} | [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] |")
    lines+=['','## Diferencias emparejadas: Double menos DQN','',
            'Cada diferencia usa la misma semilla y muestra de evaluación. No se combinan las dos muestras.','',
            '| Muestra | Diferencia media | IC 95% |','|---|---:|---:|']
    for r in paired: lines.append(f"| {r['sample']} | {r['mean']:+.2f} | [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] |")
    lines+=['','Con tres semillas estos IC dependen fuertemente del supuesto de normalidad. La comparación es exploratoria y posterior al desarrollo; no constituye una prueba confirmatoria universal.',
            'En la confirmación el IC de la diferencia incluye cero. Los seis modelos superan individualmente 200 en ambas muestras; ese hecho descriptivo no depende de que el IC entre algoritmos excluya cero.',
            'Método y límites: report/intervalos_confianza.md. No se ejecutó ningún entrenamiento para producir estas figuras.','']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__': main()
