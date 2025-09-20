#!/bin/bash

# Autonomous FinTech System Deployment Script for Digital Ocean

set -e

echo "🚀 Starting deployment of Autonomous Financial Risk Management System..."

# Configuration
DROPLET_NAME="fintech-system"
REGION="nyc1"
SIZE="s-4vcpu-8gb"
IMAGE="ubuntu-22-04-x64"
SSH_KEY_NAME="your-ssh-key"
DOMAIN="your-domain.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    print_error "doctl CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install it first."
    exit 1
fi

# Step 1: Create Digital Ocean Droplet
print_status "Creating Digital Ocean Droplet..."
DROPLET_ID=$(doctl compute droplet create $DROPLET_NAME \
    --region $REGION \
    --size $SIZE \
    --image $IMAGE \
    --ssh-keys $SSH_KEY_NAME \
    --format ID \
    --no-header \
    --wait)

print_status "Droplet created with ID: $DROPLET_ID"

# Get droplet IP address
print_status "Getting droplet IP address..."
DROPLET_IP=$(doctl compute droplet get $DROPLET_ID --format PublicIPv4 --no-header)
print_status "Droplet IP: $DROPLET_IP"

# Wait for droplet to be ready
print_status "Waiting for droplet to be ready..."
sleep 60

# Step 2: Setup server environment
print_status "Setting up server environment..."

# Copy files to server
scp -r -o StrictHostKeyChecking=no . root@$DROPLET_IP:/opt/fintech-system/

# Install dependencies and configure server
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << 'EOF'
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Node.js for frontend
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Create application directory
mkdir -p /opt/fintech-system
cd /opt/fintech-system

# Set up environment variables
cp .env.example .env

# Install SSL certificate (Let's Encrypt)
apt install -y certbot python3-certbot-nginx

# Configure firewall
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

# Create logs directory
mkdir -p /opt/fintech-system/logs

# Set permissions
chmod +x scripts/*.sh
EOF

print_status "Server environment setup completed"

# Step 3: Configure environment variables
print_status "Please configure your environment variables in /opt/fintech-system/.env"
print_warning "You need to set up the following:"
echo "- Database connection strings"
echo "- API keys for data providers"
echo "- Security keys"
echo "- Domain configuration"

# Step 4: Deploy application
print_status "Deploying application..."

ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << 'EOF'
cd /opt/fintech-system

# Build and start services
docker-compose up -d --build

# Wait for services to be ready
sleep 30

# Check service status
docker-compose ps

# Setup SSL certificate
# certbot --nginx -d your-domain.com --non-interactive --agree-tos --email your-email@example.com

# Setup log rotation
cat > /etc/logrotate.d/fintech-system << 'LOGROTATE'
/opt/fintech-system/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
LOGROTATE

# Setup monitoring with systemd
cat > /etc/systemd/system/fintech-monitor.service << 'SERVICE'
[Unit]
Description=FinTech System Monitor
After=docker.service

[Service]
Type=oneshot
ExecStart=/opt/fintech-system/scripts/health_check.sh
User=root

[Install]
WantedBy=multi-user.target
SERVICE

# Setup monitoring timer
cat > /etc/systemd/system/fintech-monitor.timer << 'TIMER'
[Unit]
Description=Run FinTech Monitor every 5 minutes
Requires=fintech-monitor.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
TIMER

systemctl enable fintech-monitor.timer
systemctl start fintech-monitor.timer

EOF

print_status "Application deployment completed"

# Step 5: Setup monitoring and backups
print_status "Setting up monitoring and backups..."

ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << 'EOF'
# Setup backup script
cat > /opt/fintech-system/scripts/backup.sh << 'BACKUP'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups"
mkdir -p $BACKUP_DIR

# Backup database
docker exec fintech-system_mongo_1 mongodump --out /tmp/backup_$DATE
docker cp fintech-system_mongo_1:/tmp/backup_$DATE $BACKUP_DIR/

# Backup application data
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz /opt/fintech-system --exclude=/opt/fintech-system/node_modules

# Upload to Digital Ocean Spaces (optional)
# s3cmd put $BACKUP_DIR/* s3://your-backup-bucket/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "backup_*" -mtime +7 -delete
find $BACKUP_DIR -name "app_backup_*" -mtime +7 -delete

echo "Backup completed: $DATE"
BACKUP

chmod +x /opt/fintech-system/scripts/backup.sh

# Setup daily backup cron job
echo "0 2 * * * root /opt/fintech-system/scripts/backup.sh" >> /etc/crontab

EOF

# Step 6: Setup domain and SSL
if [ ! -z "$DOMAIN" ]; then
    print_status "Setting up domain and SSL certificate..."
    
    ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << EOF
# Update Nginx configuration with domain
sed -i 's/your-domain.com/$DOMAIN/g' /opt/fintech-system/nginx.conf

# Restart Nginx
docker-compose restart nginx

# Get SSL certificate
certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
EOF
fi

# Step 7: Create managed databases
print_status "Setting up managed databases..."

# Create MongoDB cluster
MONGODB_CLUSTER_ID=$(doctl databases create fintech-mongodb \
    --engine mongodb \
    --region $REGION \
    --size db-s-1vcpu-1gb \
    --num-nodes 1 \
    --format ID \
    --no-header)

print_status "MongoDB cluster created with ID: $MONGODB_CLUSTER_ID"

# Create Redis cluster
REDIS_CLUSTER_ID=$(doctl databases create fintech-redis \
    --engine redis \
    --region $REGION \
    --size db-s-1vcpu-1gb \
    --num-nodes 1 \
    --format ID \
    --no-header)

print_status "Redis cluster created with ID: $REDIS_CLUSTER_ID"

print_status "Waiting for databases to be ready..."
sleep 300  # Wait 5 minutes for databases to initialize

# Get database connection details
MONGODB_URI=$(doctl databases connection fintech-mongodb --format URI --no-header)
REDIS_URI=$(doctl databases connection fintech-redis --format URI --no-header)

# Update environment variables on server
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << EOF
cd /opt/fintech-system
sed -i 's|MONGODB_URI=.*|MONGODB_URI=$MONGODB_URI|' .env
sed -i 's|REDIS_URL=.*|REDIS_URL=$REDIS_URI|' .env

# Restart services with new database connections
docker-compose restart
EOF

# Step 8: Final verification
print_status "Performing final verification..."

sleep 30

# Check if services are running
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << 'EOF'
cd /opt/fintech-system

echo "=== Service Status ==="
docker-compose ps

echo "=== Application Logs ==="
docker-compose logs --tail=50 app

echo "=== Health Check ==="
curl -f http://localhost:8000/health || echo "Health check failed"
EOF

# Display completion information
echo
echo "=========================================="
echo "🎉 Deployment completed successfully!"
echo "=========================================="
echo
echo "📊 Your Autonomous Financial Risk Management System is now running at:"
echo "🌐 IP Address: http://$DROPLET_IP"
if [ ! -z "$DOMAIN" ]; then
    echo "🌐 Domain: https://$DOMAIN"
fi
echo
echo "📋 Next Steps:"
echo "1. Configure your API keys in /opt/fintech-system/.env"
echo "2. Update database connection strings with the managed database URIs"
echo "3. Set up your domain's DNS to point to $DROPLET_IP"
echo "4. Configure SSL certificates for your domain"
echo "5. Set up monitoring and alerting"
echo
echo "🔑 Important Information:"
echo "- MongoDB Cluster ID: $MONGODB_CLUSTER_ID"
echo "- Redis Cluster ID: $REDIS_CLUSTER_ID"
echo "- Server IP: $DROPLET_IP"
echo "- SSH Access: ssh root@$DROPLET_IP"
echo
echo "📖 Documentation:"
echo "- API Documentation: http://$DROPLET_IP:8000/docs"
echo "- System Status: http://$DROPLET_IP:8000/health"
echo
echo "⚠️  Security Reminders:"
echo "1. Change default passwords"
echo "2. Set up firewall rules"
echo "3. Enable 2FA for Digital Ocean account"
echo "4. Regularly update system packages"
echo "5. Monitor system logs"
echo
print_status "Deployment script completed!"
"""
