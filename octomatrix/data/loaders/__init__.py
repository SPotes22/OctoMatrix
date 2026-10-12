"""Un loader por fuente. Cada `load(...)` devuelve un iterador de HttpSample.

cicids2018 es la excepción: produce flujos de red con su propio esquema
(experto de flujos), no peticiones HTTP.
"""
