#!/bin/bash

# Autonomous Fintech System - Complete Setup Script
# This script automates the entire deployment process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root (use sudo)"
    exit 1
fi

print_status "Starting Autonomous Fintech System setup..."

# Step 1: Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Step 2: Install required packages
print_status "Installing required packages..."
apt install -y \
    git \
    docker.io \
    docker-compose \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw \
    curl \
    htop \
    vim \
    build-essential

# Step 3: Enable Docker
print_status "Configuring Docker..."
systemctl enable docker
systemctl start docker
usermod -aG docker $USER

# Step 4: Configure swap (for ML models)
print_status "Configuring swap space..."
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
    print_status "Swap space configured"
else
    print_warning "Swap file already exists"
fi

# Step 5: Configure Git
print_status "Configuring Git..."
git config --global user.email "savankumargoswami@gmail.com"
git config --global user.name "Savan Kumar Goswami"

# Step 6: Generate SSH key for GitHub
print_status "Setting up SSH key for GitHub..."
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -C "savankumargoswami@gmail.com" -N "" -f ~/.ssh/id_ed25519
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    
    print_status "SSH Key generated. Add the following public key to your GitHub account:"
    echo ""
    cat ~/.ssh/id_ed25519.pub
    echo ""
    read -p "Press Enter after adding the key to GitHub..."
    
    # Test GitHub connection
    ssh -T git@github.com || true
else
    print_warning "SSH key already exists"
fi

# Step 7: Configure firewall
print_status "Configuring firewall..."
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 3000/tcp
ufw allow 5000/tcp
ufw allow 8080/tcp
ufw --force enable

# Step 8: Stop conflicting services
print_status "Stopping conflicting services..."
systemctl stop nginx || true
systemctl disable nginx || true
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# Step 9: Clone repository
print_status "Cloning repository..."
cd ~
if [ -d "autonomous-fintech-system" ]; then
    print_warning "Repository already exists. Pulling latest changes..."
    cd autonomous-fintech-system
    git pull origin main
else
    git clone git@github.com:savankumargoswami/autonomous-fintech-system.git
    cd autonomous-fintech-system
fi

# Step 10: Create environment file
print_status "Creating environment configuration..."
cat > .env << 'EOL'
# Database
MONGODB_URI=mongodb+srv://savangoswami0503:Gabber@9850@cluster0.tiloke1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
REDIS_URL=redis://default:password@redis-14892.c17.us-east-1-4.ec2.redns.redis-cloud.com:14892

# API Keys
ALPHA_VANTAGE_API_KEY=RTCD64TAB13Q7ARV
POLYGON_API_KEY=Wj3TIukaNewvEUpLUxn6RXJLiyo233a4
FINNHUB_API_KEY=d3459kpr01qqt8snjen0d3459kpr01qqt8snjeng
NEWS_API_KEY=4f86aa4e-b9cd-4f8d-9bab-e7e5efb43284
QUANDL_API_KEY=zbsUSsJa3CYZbry_qugu
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAABmTzwEAAAAAstjd1NbP3HwtQienqDisRviHZGA%3DJVnroOuxm47fPdbYerE9OUGedZXs3aO4ja8lVZbUNDmbQEaM9M

# Security
SECRET_KEY=HbO_YvBUQwd4unh69fqMgfro6hjt2ViNGZTTXGntY2xZnB3obZ14QeLY85CeE2hi6Ey3IAuqtoGsXNwr-tWfFQ
JWT_SECRET_KEY=your_jwt_secret_key_here_change_this

# Application
FLASK_ENV=production
NODE_ENV=production
REACT_APP_API_URL=http://$(curl -s ifconfig.me):5000
EOL

# Step 11: Create necessary directories
print_status "Creating directory structure..."
mkdir -p backend/{models,routes,services,agents,utils}
mkdir -p frontend/{public,src/{components,pages,services}}
mkdir -p ml_models/trained_models
mkdir -p logs
mkdir -p data

# Step 12: Set permissions
chmod +x setup.sh
chmod 600 .env

# Step 13: Build and start Docker containers
print_status "Building Docker images..."
docker-compose down
docker-compose build --no-cache

print_status "Starting services..."
docker-compose up -d

# Step 14: Wait for services to be ready
print_status "Waiting for services to start..."
sleep 30

# Step 15: Check service health
print_status "Checking service health..."
docker-compose ps

# Test backend
if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
    print_status "Backend is running ✓"
else
    print_error "Backend health check failed"
fi

# Test frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_status "Frontend is running ✓"
else
    print_warning "Frontend may take a few more seconds to start"
fi

# Step 16: Setup monitoring
print_status "Setting up monitoring..."
cat > monitor.sh << 'EOL'
#!/bin/bash
# Service monitoring script
docker-compose ps
echo ""
echo "=== Service Logs ==="
docker-compose logs --tail=20
echo ""
echo "=== System Resources ==="
docker stats --no-stream
EOL
chmod +x monitor.sh

# Step 17: Create backup script
print_status "Creating backup script..."
cat > backup.sh << 'EOL'
#!/bin/bash
# Backup script
BACKUP_DIR="/backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup data
docker-compose exec -T backend mongodump --uri="$MONGODB_URI" --out=$BACKUP_DIR/mongodb
docker-compose exec -T redis-local redis-cli BGSAVE

echo "Backup completed to $BACKUP_DIR"
EOL
chmod +x backup.sh

# Step 18: Setup systemd service for auto-restart
print_status "Setting up systemd service..."
cat > /etc/systemd/system/fintech-system.service << 'EOL'
[Unit]
Description=Autonomous Fintech Trading System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/root/autonomous-fintech-system
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload
systemctl enable fintech-system

# Step 19: Display summary
print_status "Setup completed successfully!"
echo ""
echo "======================================"
echo "   AUTONOMOUS FINTECH SYSTEM READY    "
echo "======================================"
echo ""
echo "Access URLs:"
echo "  Frontend:    http://$(curl -s ifconfig.me):3000"
echo "  Backend API: http://$(curl -s ifconfig.me):5000"
echo "  Nginx Proxy: http://$(curl -s ifconfig.me):8080"
echo ""
echo "Default Credentials:"
echo "  Username: demo"
echo "  Password: Demo@123456"
echo ""
echo "Useful Commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Monitor system:   ./monitor.sh"
echo "  Backup data:      ./backup.sh"
echo "  Restart services: docker-compose restart"
echo "  Stop services:    docker-compose down"
echo ""
echo "Next Steps:"
echo "1. Access the frontend and register a new account"
echo "2. Start paper trading with $100,000 virtual balance"
echo "3. Monitor your portfolio performance"
echo "4. Use AI-driven market analysis for trading decisions"
echo ""
print_warning "Remember to:"
print_warning "- Change JWT_SECRET_KEY in .env file"
print_warning "- Setup SSL certificate for production"
print_warning "- Configure domain name (optional)"
echo ""
