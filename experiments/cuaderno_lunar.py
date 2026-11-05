"""Entrada compartida por los notebooks de Tier 2 y Tier 3."""

from html import escape
import json
from pathlib import Path

import numpy as np
import torch

from experiments.lunarlander import ROOT, configuration, plan, load_model, make_env
from experiments.analizar_tier23 import available

OUTPUT = ROOT / "results/tier23_lunarlander"


def resumen(tier=2, output=OUTPUT):
    from IPython.display import HTML, display
    config=json.loads((Path(output)/"protocol.json").read_text(encoding="utf-8")) if (Path(output)/"protocol.json").exists() else configuration()
    rows=available(Path(output),config)
    rows=[r for r in rows if (tier==2 and r["group"]!="double") or (tier==3 and r["group"] in ("base","double"))]
    expected=6 if tier==2 else 2
    print(f'Tier {tier}: {len(rows)}/{expected} grupos completos; cada uno debe incluir tres semillas.')
    if not rows:
        print('Todavía no hay tres semillas completas para mostrar resultados finales.')
        return []
    body=''.join(f"<tr><td>{escape(r['group'])}</td><td>{r['final_mean']:.2f} ± {r['std_across_seed_means']:.2f}</td>"
        f"<td>{r['successful_seeds']}/3</td><td>{r['training_seconds_mean']:.1f} s</td></tr>" for r in rows)
    display(HTML('<table><tr><th>Grupo</th><th>Media final ± DE</th><th>Semillas ≥200</th><th>Entrenamiento</th></tr>'+body+'</table>'))
    return rows


def figura(name, output=OUTPUT):
    from IPython.display import Image, display
    path=Path(output)/'figures'/f'{name}.png'
    if path.exists():
        display(Image(filename=str(path)))
    else:
        print(f'Figura pendiente: {name}. Se genera al completar las tres semillas de cada tratamiento.')


def discusion(tier, output=OUTPUT):
    from IPython.display import Markdown, display
    path=Path(output)/f'report_tier{tier}.md'
    if path.exists():
        display(Markdown(path.read_text(encoding='utf-8')))
    else:
        print('La discusión se genera desde resultados completos, no desde valores supuestos.')


def demostrar(group='base',seed=42,output=None,inline=True,save_gif=False):
    """Primer episodio del protocolo, sin aprender ni buscar un intento exitoso."""
    from PIL import Image,ImageDraw
    if output is None:
        from experiments.cuaderno_mejora import active_root
        output=active_root()
    folder=Path(output)/group/f'seed_{seed}'
    if not (folder/'config.json').exists():
        raise FileNotFoundError(f'El modelo final todavía no está disponible: {folder}')
    metadata=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if metadata.get('model_file')=='online_selected.pt':
        summary_path=folder/'summary.json'
        if not summary_path.exists() or not json.loads(summary_path.read_text(encoding='utf-8')).get('complete',False):
            raise RuntimeError('La mejora sigue en validación. Su demostración de prueba se habilita después de completar las dos evaluaciones nuevas.')
        from experiments.refinar_lunar import load_selected
        model,run_config=load_selected(folder)
    else:
        model,run_config=load_model(folder)
    config=run_config['protocol']
    base=config.get('demonstration_base',config['evaluation_base'])
    env=make_env(config,render_mode='rgb_array' if inline or save_gif else 'human')
    rng=np.random.default_rng(base+500_000+1000*seed)
    frames=[]
    total=0.
    step=0
    try:
        obs,_=env.reset(seed=base+1000*seed)
        while True:
            if (inline or save_gif) and step%2==0:
                frames.append(env.render())
            with torch.inference_mode():
                values=model(torch.as_tensor(obs,dtype=torch.float32)).numpy()
            action=int(rng.choice(np.flatnonzero(values==values.max())))
            obs,reward,terminated,truncated,_=env.step(action)
            step+=1
            total+=reward
            if terminated or truncated:
                if inline or save_gif:
                    frames.append(env.render())
                break
    finally:
        env.close()
    result={'group':group,'training_seed':seed,'environment_seed':base+1000*seed,
            'return':float(total),'steps':step,'terminated':bool(terminated),'truncated':bool(truncated),
            'selection':'Primer episodio de prueba, checkpoint elegido por validación' if metadata.get('model_file') else 'Primer episodio de evaluación, no mejor intento',
            'model_file':metadata.get('model_file','online_final.pt'),'learning':False}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if save_gif:
        dest=folder/'visual'
        dest.mkdir(exist_ok=True)
        images=[]
        for frame in frames:
            image=Image.fromarray(frame)
            canvas=Image.new('RGB',(image.width,image.height+35),'#edf2f7')
            canvas.paste(image,(0,35))
            ImageDraw.Draw(canvas).text((10,10),f'{group} | semilla {seed} | sin aprendizaje',fill='#172b4d')
            # GIF usa paleta: evitar retener además todas las imágenes RGB con título.
            images.append(canvas.quantize(colors=256))
        if not inline:
            frames.clear()
        images[0].save(dest/'lander.gif',save_all=True,append_images=images[1:],duration=[40]*(len(images)-1)+[1200],loop=0)
        images[min(80,len(images)-1)].save(dest/'vista_previa.png')
        (dest/'demostracion.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if inline:
        from IPython.display import HTML,display
        from matplotlib import animation
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(6,4))
        FigureCanvasAgg(fig)
        ax=fig.subplots()
        artist=ax.imshow(frames[0])
        ax.axis('off')
        def update(index):
            artist.set_data(frames[index])
            return (artist,)
        anim=animation.FuncAnimation(fig,update,frames=len(frames),interval=40,blit=True)
        html=anim.to_jshtml(embed_frames=True,default_mode='once')
        fig.clear()
        display(HTML(html))
    return result
