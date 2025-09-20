#!/bin/bash

# Health check script for monitoring system components

LOGFILE="/opt/fintech-system/logs/health_check.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

log_message() {
    echo "[$DATE] $1" >> $LOGFILE
    echo "[$DATE] $1"
}

# Check if Docker containers are running
check_containers() {
    log_message "Checking Docker containers..."
    
    cd /opt/fintech-system
    
    # Check if containers are running
    if docker-compose ps | grep -q "Up"; then
        log_message "✅ Docker containers are running"
        return 0
    else
        log_message "❌ Some Docker containers are not running"
        docker-compose ps >> $LOGFILE
        return 1
    fi
}

# Check application health endpoint
check_application() {
    log_message "Checking application health..."
    
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log_message "✅ Application health check passed"
        return 0
    else
        log_message "❌ Application health check failed"
        return 1
    fi
}

# Check database connections
check_databases() {
    log_message "Checking database connections..."
    
    # Check MongoDB
    if docker exec fintech-system_mongo_1 mongo --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        log_message "✅ MongoDB is responsive"
    else
        log_message "❌ MongoDB connection failed"
    fi
    
    # Check Redis
    if docker exec fintech-system_redis_1 redis-cli ping > /dev/null 2>&1; then
        log_message "✅ Redis is responsive"
    else
        log_message "❌ Redis connection failed"
    fi
}

# Check system resources
check_resources() {
    log_message "Checking system resources..."
    
    # Check disk space
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 90 ]; then
        log_message "⚠️  High disk usage: ${DISK_USAGE}%"
    else
        log_message "✅ Disk usage is normal: ${DISK_USAGE}%"
    fi
    
    # Check memory usage
    MEMORY_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
        log_message "⚠️  High memory usage: ${MEMORY_USAGE}%"
    else
        log_message "✅ Memory usage is normal: ${MEMORY_USAGE}%"
    fi
    
    # Check CPU load
    CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    log_message "📊 Current CPU load: $CPU_LOAD"
}

# Restart services if needed
restart_if_needed() {
    if ! check_containers || ! check_application; then
        log_message "🔄 Attempting to restart services..."
        
        cd /opt/fintech-system
        docker-compose restart
        
        sleep 30
        
        if check_containers && check_application; then
            log_message "✅ Services restarted successfully"
        else
            log_message "❌ Service restart failed - manual intervention required"
            # Send alert (implement your preferred alerting method)
        fi
    fi
}

# Main execution
log_message "Starting health check..."

check_containers
check_application
check_databases
check_resources
restart_if_needed

log_message "Health check completed"

# Cleanup old logs (keep last 30 days)
find /opt/fintech-system/logs -name "*.log" -mtime +30 -delete
"""
