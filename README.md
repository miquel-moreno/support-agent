# Asistente que responde emails de clientes, con aprobación humana

**EN** · An AI agent that reads customer emails, looks up the order, checks the return policy and drafts a reply that a person approves before it is sent.

Para tiendas y talleres que reciben cada día correos como "¿dónde está mi pedido?" o "quiero devolverlo" y los contestan uno a uno.

> 🚧 **En desarrollo.** Primera versión prevista para octubre de 2026.

## Qué hace
- Entiende qué pide el cliente: estado del pedido, devolución, problema con la factura u otra consulta
- Consulta el pedido y comprueba la política de devoluciones antes de responder
- Redacta la respuesta y una persona la aprueba antes de enviarla
- Nunca se inventa datos de un pedido ni promete lo que la política no permite

## Resultado
- En 40 emails de prueba entiende qué pide el cliente y encuentra el pedido correcto en los 40, sin inventarse ningún dato
- 39 de 40 respuestas se podrían enviar sin cambios (revisión automática, comprobada a mano en una muestra)
- Contesta cada email en unos 3 segundos por menos de 0,1 céntimos

## Tecnologías
Python · FastAPI · IA (agentes, LangGraph, LangChain) · PostgreSQL · Docker · GitHub Actions

## Mi papel
Lo he diseñado y desarrollado de principio a fin. Desarrollo asistido por IA bajo mi especificación y revisión.

[Detalles técnicos →](docs/TECNICO.md) · [LinkedIn](https://www.linkedin.com/in/miquel-moreno-martinez)
