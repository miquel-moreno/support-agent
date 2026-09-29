# Detalles técnicos

## Ponerlo en marcha

```bash
cp .env.example .env   # elige proveedor de LLM y añade tu clave
docker compose up --build
```

La API queda en http://localhost:8000 (documentación interactiva en `/docs`).

Desarrollo local:

```bash
make install   # dependencias + hooks de pre-commit
make dev       # API con recarga automática
make check     # lint + tipos + tests
make eval      # evaluación con un LLM real (solo en local)
```

## Datos de prueba: la tienda

Todo es inventado (Faker `es_ES`, semilla fija): 30 clientes con emails `example.*` (nunca buzones reales) y 80 pedidos en todos los estados (pendiente, enviado, entregado, cancelado, devuelto), con fechas, transportista, productos, total y número de factura. La fecha de referencia de la tienda es fija (28/09/2026) para que la política dé siempre el mismo resultado.

```bash
uv run python -m scripts.seed_shop              # carga (o reemplaza) la tienda
uv run python -m scripts.seed_shop --if-empty   # solo si está vacía
```

## Arquitectura

```
src/support_agent/
├── api/        # rutas HTTP (FastAPI) y middleware
├── core/       # configuración, logging JSON, errores
├── services/   # agente (grafo), herramientas, política y revisión de borradores
└── adapters/   # LLM (LangChain) y base de datos
```

### El grafo del agente (LangGraph)

```
classify ─┬─ other ──────────────────────────────────────────┐
          └─ lookup ⇄ tools → resolve → policy → draft → review ─┬─ end
                                            ▲                     │
                                            └── 1 reescritura ────┘
```

| Nodo | Quién decide | Qué hace |
|---|---|---|
| `classify` | LLM (salida estructurada) | `order_status`, `return_request`, `invoice_problem` u `other` |
| `lookup` ⇄ `tools` | LLM (tool calling) | Elige `find_order(número)` o `find_my_orders()`; máximo 3 vueltas |
| `resolve` | Código | Con qué pedido nos quedamos: encontrado, elegir entre varios, pedir el número o pedido ajeno |
| `policy` | Código | `check_return()` si es una devolución |
| `draft` | LLM | Borrador en español solo con los datos recibidos |
| `review` | Código | Todo número de pedido, seguimiento, factura, fecha e importe del borrador debe existir en los datos |

Probarlo con un email (tienda en SQLite en memoria, LLM del `.env`):

```bash
uv run python -m scripts.try_agent --order PED-10001 "Quiero devolver el pedido PED-10001"
```

## Decisiones técnicas

| Decisión | Por qué |
|---|---|
| La política de devoluciones está en código (`services/policy.py`), no en el prompt | Que el agente pueda prometer un reembolso no depende de que el modelo lea bien las instrucciones: el código decide (30 días desde la entrega, cancelar si no se ha enviado, esperar si está en camino) y el agente solo lo explica. Cada regla tiene su test |
| Fecha de referencia fija en los datos de prueba | La política depende de "hoy"; con una fecha fija, la evaluación da el mismo resultado cualquier día |
| LLM con `ChatOpenAI` de LangChain, elegido por `LLM_PROVIDER` | Soporta tool calling y salida estructurada; OpenAI y Ollama hablan la misma API, así que cambiar de proveedor es cambiar el `.env`. Los tests usan `ScriptedChatModel` (respuestas guionizadas): el CI no gasta dinero ni necesita claves |
| LangGraph en vez de un bucle hecho a mano | El flujo queda explícito (nodos y aristas), con un tope de vueltas, y el checkpointer y la interrupción para la aprobación humana vienen de serie |
| Las herramientas solo devuelven filas reales | El modelo no puede inventarse un pedido: lo que escribe sobre él viene de la base de datos. El email del remitente lo inyecta el grafo (`InjectedState`), el modelo no lo ve ni lo elige |
| Un pedido de otro email no se revela | Privacidad: si alguien pregunta por un número que no es suyo, la herramienta no devuelve ningún dato y se le pide que escriba desde su email |
| `find_order` solo busca números escritos en el email | En una prueba real, `qwen2.5:3b` se inventó un número cuando el email no traía ninguno. Se impide con código, no con el prompt |
| Revisión del borrador con código y una reescritura | Si el borrador trae un dato que no está en los hechos, el modelo lo reescribe una vez; si persiste, el aviso queda para la persona que aprueba |

## Evaluación

_Pendiente._ Resultados en `evals/results/`, con fecha y modelo.

## Limitaciones

- La revisión con código detecta datos inventados (números, fechas, importes, códigos), pero no frases inventadas: en una prueba real el modelo añadió un motivo de rechazo que nadie había dado. Por eso una persona aprueba cada respuesta.
- Con un modelo local pequeño (`qwen2.5:3b`) los datos son correctos pero la redacción es torpe y cada email tarda ~1 minuto en CPU.
