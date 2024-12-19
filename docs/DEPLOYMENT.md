# Trading Bot Deployment Guide

## System Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 2 CPU cores minimum
- 20GB disk space

## Quick Start

1. Clone the repository:
```bash
git clone <repository_url>
cd trading_bot
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Build and start services:
```bash
docker-compose -f deployment/docker-compose.yml up -d
```

4. Verify deployment:
```bash
docker-compose -f deployment/docker-compose.yml ps
```

## Service Architecture

The system consists of three main services:

1. Trading Bot (Port 8000)
   - Main trading logic
   - API endpoints
   - WebSocket connections
   - Risk management

2. Redis (Port 6379)
   - Data caching
   - Real-time market data
   - Trading state management

3. Monitoring Stack (Ports 9090, 3000)
   - Prometheus metrics
   - Grafana dashboards
   - System health monitoring

## Configuration

### Trading Bot Configuration

Edit `config/trading_config.yml`:
```yaml
exchange:
  name: binance
  api_key: ${EXCHANGE_API_KEY}
  api_secret: ${EXCHANGE_API_SECRET}

trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  timeframe: "1m"
  max_position_size: 0.1
  stop_loss_pct: 0.02

risk:
  max_drawdown: 0.1
  max_trades_per_day: 10
  max_position_value: 1000
```

### Monitoring Configuration

1. Prometheus (`deployment/config/prometheus.yml`)
   - Metrics collection interval
   - Storage retention
   - Alert rules

2. Grafana (`deployment/config/grafana.ini`)
   - Dashboard configuration
   - Data source setup
   - User authentication

## Deployment Steps

1. **Pre-deployment Checks**
   - Verify system requirements
   - Check network connectivity
   - Validate configuration files

2. **Initial Deployment**
   ```bash
   # Build images
   docker-compose -f deployment/docker-compose.yml build

   # Start services
   docker-compose -f deployment/docker-compose.yml up -d
   ```

3. **Post-deployment Verification**
   ```bash
   # Check service logs
   docker-compose -f deployment/docker-compose.yml logs -f

   # Verify endpoints
   curl http://localhost:8000/health
   ```

## Monitoring

1. Access Grafana:
   - URL: http://localhost:3000
   - Default credentials: admin/admin

2. Access Prometheus:
   - URL: http://localhost:9090

3. Key Metrics:
   - Trading performance
   - System resources
   - Error rates
   - Latency

## Maintenance

1. **Backup**
   ```bash
   # Backup data volumes
   docker run --rm -v trading_bot_redis_data:/data -v /backup:/backup ubuntu tar czf /backup/redis-backup.tar.gz /data
   ```

2. **Updates**
   ```bash
   # Pull latest changes
   git pull

   # Rebuild and restart services
   docker-compose -f deployment/docker-compose.yml down
   docker-compose -f deployment/docker-compose.yml build
   docker-compose -f deployment/docker-compose.yml up -d
   ```

3. **Logs**
   ```bash
   # View service logs
   docker-compose -f deployment/docker-compose.yml logs -f trading_bot
   docker-compose -f deployment/docker-compose.yml logs -f redis
   docker-compose -f deployment/docker-compose.yml logs -f monitoring
   ```

## Troubleshooting

1. **Service Fails to Start**
   - Check logs: `docker-compose logs [service_name]`
   - Verify configuration
   - Check resource availability

2. **Performance Issues**
   - Monitor resource usage
   - Check network connectivity
   - Review log files

3. **Data Issues**
   - Verify Redis connection
   - Check disk space
   - Validate data integrity

## Security Considerations

1. **API Keys**
   - Use environment variables
   - Rotate regularly
   - Limit API permissions

2. **Network Security**
   - Use internal Docker network
   - Limit exposed ports
   - Enable TLS where possible

3. **Access Control**
   - Change default credentials
   - Implement role-based access
   - Regular security audits

## Support

For issues and support:
1. Check the troubleshooting guide
2. Review logs and metrics
3. Contact support team
4. Create GitHub issue 