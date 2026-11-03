# Auditoría del flujo y cálculo DQN

## Conclusión

No se identificó un error en los aspectos comprobados del flujo de experiencias,
los objetivos, la pérdida o la dirección de actualización. El agente también
resolvió un problema pequeño con valores óptimos conocidos. Esto reduce la
sospecha de un fallo básico, pero no demuestra que todo el programa esté libre
de errores ni explica por sí solo el colapso en CartPole.
No se corrigió el agente ni se modificaron retrospectivamente sus resultados.

## Fuentes primarias consultadas

Consulta: 2 de noviembre de 2026. La rama master puede cambiar posteriormente.

- [Fuente DQN de CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py).
- [Documentación DQN de CleanRL](https://docs.cleanrl.dev/rl-algorithms/dqn/).
- [Buffer de CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/buffers.py).
- [Gymnasium: límites temporales](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

La fórmula de objetivo y la selección del valor correspondiente a la acción
tomada coinciden con la estructura de la fuente DQN de CleanRL. Nuestro agente
se implementó por el equipo; aquí se revisó la referencia, no se ejecutó un
agente prefabricado ni se atribuyeron a nuestro código resultados publicados.

Gymnasium distingue fin real de corte externo: se elimina el término futuro
solo ante `terminated`. Nuestro entorno no es vectorizado y se reinicia
manualmente, de modo que la observación posterior al paso se guarda antes del
reset. La referencia vectorizada necesita tratar por separado la observación
final cuando el entorno realiza reinicios automáticos.

## Diferencias de configuración observadas

Valores leídos en la fuente CleanRL consultada, no un resultado propio:

| Aspecto | Primer DQN del proyecto | Fuente CleanRL consultada |
|---|---|---|
| Capas internas | 64, 64 | 120, 84 |
| Tasa | 0.0005 | 0.00025 |
| Memoria | 50 000 | 10 000 |
| Lote | 64 | 128 |
| Inicio de ajustes | 1 000 pasos | Más de 10 000 pasos |
| Ajuste | Cada 4 pasos | Cada 10 pasos |
| Decaimiento de exploración | 80 % del presupuesto | 50 % del presupuesto |
| Pérdida | Smooth L1 | MSE |
| Límite de gradiente | Norma 10 | No aplicado en el bloque de actualización consultado |
| Presupuesto | 200 000 pasos | 500 000 pasos por defecto |

Son variantes de diseño, no errores demostrados. La documentación y el código
pueden diferir: el ejemplo documental menciona un límite de gradiente que no
aparece aplicado en el bloque consultado. No se cambiaron todos estos parámetros
para presentar un resultado como si variara una sola cosa.

## Auditoría reproducible propia

Código: `experiments/auditar_dqn.py`. Evidencia: `results/parte_13_auditoria/audit.json`.

### Experiencias reales y reinicios

Se ejecutaron 600 pasos de CartPole con límite externo de 17 pasos para provocar
ambos finales: se observaron 13 terminaciones y 27 truncamientos. La memoria
circular tuvo capacidad 128. Se recuperaron los 128 recuerdos conservados y
se comparó cada observación, acción, recompensa, siguiente observación y bandera
con el registro original. Todos coincidieron.
Una comprobación aparte con límite de un paso confirmó que el reset posterior
no sustituye la observación final guardada en la memoria.

### Cálculos independientes

Se reconstruyeron las capas lineales y ReLU con NumPy en precisión de 64 bits,
sin usar el forward de PyTorch para los resultados de referencia. Con un lote
real se verificaron los objetivos y la expresión por tramos de Smooth L1.

- Máxima diferencia absoluta en los objetivos: 8.06e-8.
- Pérdida PyTorch: 0.5973546505; cálculo NumPy: 0.5973546627.
- Derivada por diferencias finitas de un sesgo: −0.4842158392.
- Gradiente correspondiente de PyTorch: −0.4842158556.
- Norma del gradiente: 0.73345; el límite 10 no se activó en ese caso.

La comprobación por diferencias finitas cubre un parámetro y un lote; no es una
demostración de todas las derivadas posibles.

### Control con solución conocida

Se construyó un MDP de dos estados que alternan. En cada estado, una acción da
recompensa 1 y la otra 0. Con gamma=0.9, los valores óptimos son 10 para la
acción correcta y 9 para la incorrecta. Se guardaron las cuatro transiciones
posibles y se entrenó el mismo agente sobre ellas, con replay y red objetivo.

Se usaron 4 000 ajustes por semilla, red interna 16-16, tasa 0.001 y copia cada
100 ajustes. Es una prueba de control distinta de CartPole, con criterios fijados
en el script: acciones correctas y error absoluto máximo menor que 0.5.

| Semilla | Error máximo frente a Q óptimo | Acciones elegidas en ambos estados |
|---|---:|---|
| 42 | 0.1521 | Correctas |
| 43 | 0.1447 | Correctas |
| 44 | 0.1590 | Correctas |

El agente aprendió a distinguir situaciones en este problema simple. No se
presenta esto como éxito en CartPole ni como prueba de estabilidad general.

## Reproducción y siguiente experimento

```text
python -m experiments.auditar_dqn
```

El comando falla si alguna comprobación no se cumple y guarda el registro solo
al terminar las tres pruebas de control. No reentrena CartPole ni modifica sus
modelos guardados. No añadió dependencias.

La diferencia de pérdida, Smooth L1 frente a MSE, merece un estudio aislado:
las funciones dan distinto peso a errores grandes. Esto es una hipótesis de
configuración, no una causa confirmada del colapso. El siguiente experimento
puede comparar ambas pérdidas con iguales pesos iniciales, recuerdos, semillas
y presupuesto, conservando los demás parámetros y todos los resultados previos.
