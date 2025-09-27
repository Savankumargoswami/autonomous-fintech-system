# Autonomous Financial Risk Management Ecosystem

A sophisticated multi-agent trading and risk management system that autonomously manages portfolios, detects market anomalies, and adapts to changing market conditions.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-18.2+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-green.svg)

## 🚀 Features

### Core Trading Features
- **Paper Trading**: Practice trading with $100,000 virtual balance
- **Real-time Market Data**: Live price feeds from multiple sources
- **Portfolio Management**: Track positions, P&L, and performance metrics
- **Order Types**: Market, limit, stop-loss, and take-profit orders
- **Risk Management**: Automated position sizing and risk controls

### AI-Powered Intelligence
- **Multi-Agent System**: Collaborative AI agents for different aspects of trading
  - Strategy Agent: Selects optimal trading strategies
  - Risk Agent: Monitors and manages portfolio risk
  - Execution Agent: Optimizes order execution
  - Sentiment Agent: Analyzes market sentiment
- **Machine Learning Models**:
  - Deep Reinforcement Learning (PPO, SAC, TD3)
  - Transformer models for time-series forecasting
  - Graph Neural Networks for market relationships
- **Adaptive Learning**: Continuously improves from market conditions

### Technical Analysis
- **Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Pattern Recognition**: Automated chart pattern detection
- **Market Regime Detection**: Identifies trending, ranging, volatile markets
- **Multi-timeframe Analysis**: Comprehensive market view

### User Interface
- **Modern Dashboard**: Real-time portfolio overview
- **Interactive Charts**: TradingView-style charting
- **Trade Execution Panel**: Quick order placement
- **Performance Analytics**: Detailed trading statistics
- **Mobile Responsive**: Trade from any device

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                      │
│  Dashboard | Trading | Portfolio | Analysis | Settings   │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket + REST API
┌────────────────────┴────────────────────────────────────┐
│                   Backend (Flask + SocketIO)             │
│  Authentication | Trading Engine | Risk Manager | APIs   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                      AI Layer                            │
│  Strategy Agent | Risk Agent | Execution | Sentiment     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                    Data Layer                            │
│      MongoDB | Redis | Market Data APIs | ML Models      │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask, Flask-SocketIO
- **Database**: MongoDB (Atlas), Redis
- **ML/AI**: PyTorch, TensorFlow, Stable-Baselines3
- **Data Processing**: Pandas, NumPy, SciPy

### Frontend
- **Framework**: React 18
- **State Management**: Context API + Hooks
- **Styling**: Tailwind CSS
- **Charts**: Recharts, TradingView Widgets
- **WebSocket**: Socket.io-client

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Web Server**: Nginx
- **Deployment**: Digital Ocean
- **Monitoring**: Prometheus, Grafana (optional)

## 📦 Installation

### Prerequisites
- Docker and Docker Compose
- Git
- Ubuntu 22.04 (for production deployment)
- 8GB RAM minimum
- 20GB storage

### Quick Start (Local Development)

1. **Clone the repository**
```bash
git clone git@github.com:savankumargoswami/autonomous-fintech-system.git
cd autonomous-fintech-system
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Run with Docker Compose**
```bash
docker-compose up --build
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- API Docs: http://localhost:5000/api/docs

### Production Deployment (Digital Ocean)

1. **Run the automated setup script**
```bash
sudo ./setup.sh
```

2. **Follow the prompts to**:
- Configure Git credentials
- Set up SSH keys for GitHub
- Configure environment variables
- Build and deploy containers

3. **Access your deployed application**
- Frontend: http://your-droplet-ip:3000
- Backend: http://your-droplet-ip:5000

## 🔧 Configuration

### Environment Variables
```env
# Database
MONGODB_URI=your_mongodb_connection_string
REDIS_URL=your_redis_connection_string

# API Keys
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key
FINNHUB_API_KEY=your_key
NEWS_API_KEY=your_key

# Security
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_key
```

### Trading Parameters
Edit `backend/config.py`:
- Initial balance: `DEFAULT_PAPER_TRADING_BALANCE`
- Commission rate: `DEFAULT_COMMISSION_RATE`
- Risk limits: `MAX_POSITION_SIZE`, `STOP_LOSS_PERCENTAGE`

## 📊 Usage

### Creating an Account
1. Navigate to http://localhost:3000/register
2. Enter username, email, and password
3. Start with $100,000 virtual balance

### Placing Trades
1. Go to Trading Interface
2. Search for a symbol (e.g., AAPL)
3. Select order type (Market/Limit)
4. Enter quantity
5. Click Buy/Sell

### Using AI Analysis
1. Navigate to Market Analysis
2. Enter a symbol
3. Click "Analyze"
4. View AI recommendations for:
   - Trading strategy
   - Risk assessment
   - Market sentiment
   - Entry/Exit points

### Monitoring Performance
- **Dashboard**: Real-time portfolio value
- **Portfolio**: Detailed positions and P&L
- **Analytics**: Trading statistics and metrics

## 🧪 Testing

### Run Unit Tests
```bash
docker-compose exec backend pytest tests/
```

### Run Integration Tests
```bash
docker-compose exec backend pytest tests/integration/
```

### Test Trading Strategies
```bash
docker-compose exec backend python -m scripts.backtest
```

## 📈 Performance Metrics

The system tracks:
- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Time in positions
- **Profit Factor**: Gross profit / Gross loss

## 🔒 Security

- **Authentication**: JWT-based authentication
- **Password Security**: Bcrypt hashing
- **API Rate Limiting**: Prevents abuse
- **Input Validation**: Sanitized user inputs
- **HTTPS**: SSL/TLS encryption (production)
- **Environment Variables**: Sensitive data protection

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find and kill process
sudo lsof -i :PORT_NUMBER
sudo kill -9 PID
```

**Database Connection Failed**
- Check MongoDB URI in .env
- Verify network connectivity
- Ensure IP whitelist includes your server

**Docker Build Fails**
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

**Module Import Errors**
```bash
# Rebuild containers
docker-compose down
docker-compose up --build
```

## 📝 API Documentation

### Authentication
```http
POST /api/auth/register
POST /api/auth/login
GET /api/auth/profile
```

### Trading
```http
POST /api/trading/execute
DELETE /api/trading/order/:id
GET /api/trading/orders
```

### Portfolio
```http
GET /api/portfolio
GET /api/portfolio/performance
GET /api/portfolio/transactions
```

### Market Data
```http
GET /api/market/quote/:symbol
GET /api/market/analysis/:symbol
GET /api/market/news/:symbol
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Financial data providers (Alpha Vantage, Polygon, Finnhub)
- Open-source ML libraries (PyTorch, TensorFlow, Stable-Baselines3)
- React and Flask communities

## 📧 Contact

**Savan Kumar Goswami**
- GitHub: [@savankumargoswami](https://github.com/savankumargoswami)
- Email: savankumargoswami@gmail.com

## ⚠️ Disclaimer

This is a paper trading system for educational purposes only. No real money is involved. Always do your own research and consult with financial advisors before making investment decisions.

---

**Built with ❤️ by Savan Kumar Goswami**
