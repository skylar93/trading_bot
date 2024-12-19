# Multi-stage build for monitoring services
FROM prom/prometheus:v2.45.0 as prometheus
FROM grafana/grafana:10.0.3 as grafana

# Final stage
FROM ubuntu:22.04

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy Prometheus
COPY --from=prometheus /bin/prometheus /bin/prometheus
COPY --from=prometheus /bin/promtool /bin/promtool
COPY deployment/config/prometheus.yml /etc/prometheus/prometheus.yml

# Copy Grafana
COPY --from=grafana /usr/share/grafana /usr/share/grafana
COPY deployment/config/grafana.ini /etc/grafana/grafana.ini
COPY deployment/config/dashboards /etc/grafana/dashboards

# Set up directories
RUN mkdir -p /var/lib/prometheus /var/lib/grafana /etc/monitoring
RUN chown -R nobody:nogroup /var/lib/prometheus /var/lib/grafana

# Copy supervisor configuration
COPY deployment/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 9090 3000

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 