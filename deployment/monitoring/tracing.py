"""
OpenTelemetry tracing integration (H4 — Week 78).

Replaces ad-hoc timestamp diffs with OTel spans so latency is
automatically propagated to any OTLP-compatible backend.

Span naming convention:
    trading.order.submit        – full order round-trip (submit → exchange ack)
    trading.order.risk_check    – UnifiedRiskManager pre-trade check
    trading.order.compliance    – pre-trade compliance gate (limits, wash-trade)
    trading.agent.decide        – agent.predict() call
    trading.data.feed_tick      – single market data tick processing

Usage (in OrderManager.submit_order):
    with start_span("trading.order.submit", attributes={"symbol": symbol}) as span:
        ...
        span.set_attribute("order.id", order_id)

Configuration:
    Set OTEL_EXPORTER_OTLP_ENDPOINT env var to send to a collector
    (e.g. http://localhost:4318 for Jaeger OTLP HTTP).
    Defaults to ConsoleSpanExporter if not set (prints to stdout).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    logger.info("opentelemetry-sdk not installed; tracing disabled")

_tracer: Optional[Any] = None
_provider: Optional[Any] = None


def init_tracing(
    service_name: str = "trading-bot",
    otlp_endpoint: Optional[str] = None,
) -> None:
    """Initialise the global OTel tracer.

    Call once at application startup (e.g. in PaperTrader.__init__).

    Parameters
    ----------
    service_name:
        OTel resource service.name attribute.
    otlp_endpoint:
        OTLP HTTP endpoint (e.g. http://localhost:4318).
        Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var, then ConsoleSpanExporter.
    """
    global _tracer, _provider

    if not HAS_OTEL:
        logger.warning("OTel not available — tracing disabled")
        return

    if _provider is not None:
        return  # already initialised

    resource = Resource(attributes={ResourceAttributes.SERVICE_NAME: service_name})
    _provider = TracerProvider(resource=resource)

    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
            _provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTel tracing → OTLP endpoint: %s", endpoint)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp not installed; falling back to console exporter"
            )
            _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        # Local dev: print spans to stdout so they're visible without a collector
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel tracing → ConsoleSpanExporter (set OTEL_EXPORTER_OTLP_ENDPOINT to use OTLP)")

    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer(service_name)
    logger.info("OTel tracer initialised (service=%s)", service_name)


def get_tracer() -> Any:
    """Return the global tracer (or a no-op proxy if OTel unavailable)."""
    if not HAS_OTEL:
        return _NoopTracer()
    if _tracer is None:
        return trace.get_tracer("trading-bot")
    return _tracer


@contextmanager
def start_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a code block in an OTel span.

    Safe to use even when OTel is not initialised — yields a no-op span.

    Parameters
    ----------
    name:
        Span name (e.g. "trading.order.submit").
    attributes:
        Initial span attributes dict.

    Example
    -------
    with start_span("trading.order.submit", {"symbol": "BTCUSDT"}) as span:
        result = exchange.create_order(...)
        span.set_attribute("order.id", result["id"])
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes and HAS_OTEL:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            if HAS_OTEL:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
            raise


def record_order_latency(span: Any, latency_ms: float) -> None:
    """Attach order latency to the current span as an attribute."""
    if HAS_OTEL and span is not None:
        span.set_attribute("order.latency_ms", latency_ms)


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider. Call on application exit."""
    global _provider, _tracer
    if _provider is not None and HAS_OTEL:
        _provider.shutdown()
        _provider = None
        _tracer = None


# ------------------------------------------------------------------
# No-op fallback when OTel is not installed
# ------------------------------------------------------------------

class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def set_status(self, *args: Any) -> None:
        pass

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str) -> Generator[_NoopSpan, None, None]:
        yield _NoopSpan()
