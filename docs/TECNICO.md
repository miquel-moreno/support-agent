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
├── services/   # lógica de negocio, sin red: se prueba con tests unitarios
└── adapters/   # LLM, base de datos y APIs externas, detrás de interfaces
```

## Decisiones técnicas

| Decisión | Por qué |
|---|---|
| La política de devoluciones está en código (`services/policy.py`), no en el prompt | Que el agente pueda prometer un reembolso no depende de que el modelo lea bien las instrucciones: el código decide (30 días desde la entrega, cancelar si no se ha enviado, esperar si está en camino) y el agente solo lo explica. Cada regla tiene su test |
| Fecha de referencia fija en los datos de prueba | La política depende de "hoy"; con una fecha fija, la evaluación da el mismo resultado cualquier día |
| LLM detrás de una interfaz propia (`LLMClient`) | Cambiar de proveedor (OpenAI, Anthropic, Ollama local) es cambiar una variable de entorno. Los tests usan un cliente falso: el CI no gasta dinero ni necesita claves. |

## Evaluación

_Pendiente._ Resultados en `evals/results/`, con fecha y modelo.

## Limitaciones

_Pendiente._
