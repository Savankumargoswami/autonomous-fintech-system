import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Create root element
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the app
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

/* frontend/src/App.css */
.app {
  min-height: 100vh;
  background: #0f0f1e;
  color: #ffffff;
}

/* Loading Container */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
}

.spinner {
  width: 50px;
  height: 50px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Navbar */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: #1a1a2e;
  border-bottom: 2px solid #16213e;
}

.nav-brand h2 {
  color: #4fbdba;
  font-size: 1.5rem;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-links a {
  color: #ffffff;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 5px;
  transition: all 0.3s ease;
}

.nav-links a:hover,
.nav-links a.active {
  background: #4fbdba;
  color: #1a1a2e;
}

.nav-user {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.nav-user button {
  padding: 0.5rem 1rem;
  background: #e94560;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s ease;
}

.nav-user button:hover {
  background: #c13651;
}

/* Login/Register */
.login-container,
.register-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-box,
.register-box {
  background: white;
  padding: 2rem;
  border-radius: 10px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  width: 100%;
  max-width: 400px;
}

.login-box h2,
.register-box h2 {
  color: #333;
  margin-bottom: 1.5rem;
  text-align: center;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: #555;
  font-weight: 500;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 1rem;
  transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: #667eea;
}

button[type="submit"] {
  width: 100%;
  padding: 0.75rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 5px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.3s ease;
}

button[type="submit"]:hover {
  opacity: 0.9;
}

button[type="submit"]:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-message {
  background: #fee;
  color: #c33;
  padding: 0.75rem;
  border-radius: 5px;
  margin-bottom: 1rem;
  text-align: center;
}

/* Dashboard */
.dashboard {
  padding: 2rem;
}

.dashboard h1 {
  color: #4fbdba;
  margin-bottom: 2rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: #1a1a2e;
  padding: 1.5rem;
  border-radius: 10px;
  border: 1px solid #16213e;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 30px rgba(79, 189, 186, 0.3);
}

.stat-card h3 {
  color: #7ec8e3;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: #ffffff;
}

.stat-value.positive {
  color: #4fbdba;
}

.stat-value.negative {
  color: #e94560;
}

/* Charts Section */
.charts-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.chart-container,
.positions-table {
  background: #1a1a2e;
  padding: 1.5rem;
  border-radius: 10px;
  border: 1px solid #16213e;
}

.chart-container h3,
.positions-table h3 {
  color: #4fbdba;
  margin-bottom: 1rem;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
}

thead {
  background: #16213e;
}

th {
  padding: 0.75rem;
  text-align: left;
  color: #7ec8e3;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.85rem;
}

td {
  padding: 0.75rem;
  color: #ffffff;
  border-bottom: 1px solid #16213e;
}

tr:hover {
  background: rgba(79, 189, 186, 0.1);
}

td.positive {
  color: #4fbdba;
}

td.negative {
  color: #e94560;
}

/* Trading Interface */
.trading-interface {
  padding: 2rem;
}

.trading-interface h2 {
  color: #4fbdba;
  margin-bottom: 2rem;
}

.trading-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.quote-section {
  background: #1a1a2e;
  padding: 2rem;
  border-radius: 10px;
  border: 1px solid #16213e;
}

.quote-section input {
  width: 100%;
  padding: 0.75rem;
  background: #0f0f1e;
  border: 1px solid #16213e;
  color: white;
  border-radius: 5px;
  font-size: 1rem;
}

.quote-display {
  margin-top: 1.5rem;
  text-align: center;
}

.quote-display h3 {
  color: #7ec8e3;
  margin-bottom: 0.5rem;
}

.quote-display .price {
  font-size: 2.5rem;
  font-weight: 700;
  color: white;
  margin-bottom: 0.5rem;
}

.order-form {
  background: #1a1a2e;
  padding: 2rem;
  border-radius: 10px;
  border: 1px solid #16213e;
}

.form-row {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.side-btn {
  flex: 1;
  padding: 1rem;
  background: #16213e;
  color: white;
  border: 2px solid transparent;
  border-radius: 5px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.side-btn.active.buy {
  background: #4fbdba;
  border-color: #4fbdba;
  color: #1a1a2e;
}

.side-btn.active.sell {
  background: #e94560;
  border-color: #e94560;
  color: white;
}

.submit-btn {
  width: 100%;
  padding: 1rem;
  border: none;
  border-radius: 5px;
  font-size: 1rem;
  font-weight: 600;
  color: white;
  cursor: pointer;
  transition: opacity 0.3s ease;
}

.submit-btn.buy {
  background: #4fbdba;
}

.submit-btn.sell {
  background: #e94560;
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.message {
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 5px;
  text-align: center;
}

.message.success {
  background: rgba(79, 189, 186, 0.2);
  color: #4fbdba;
  border: 1px solid #4fbdba;
}

.message.error {
  background: rgba(233, 69, 96, 0.2);
  color: #e94560;
  border: 1px solid #e94560;
}

/* Market Analysis */
.market-analysis {
  padding: 2rem;
}

.market-analysis h1 {
  color: #4fbdba;
  margin-bottom: 2rem;
}

.analysis-input {
  display: flex;
  gap: 1rem;
  max-width: 600px;
  margin-bottom: 2rem;
}

.analysis-input input {
  flex: 1;
  padding: 0.75rem;
  background: #1a1a2e;
  border: 1px solid #16213e;
  color: white;
  border-radius: 5px;
  font-size: 1rem;
}

.analysis-input button {
  padding: 0.75rem 2rem;
  background: #4fbdba;
  color: #1a1a2e;
  border: none;
  border-radius: 5px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.3s ease;
}

.analysis-input button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.analysis-results {
  background: #1a1a2e;
  padding: 2rem;
  border-radius: 10px;
  border: 1px solid #16213e;
}

.analysis-results h2 {
  color: #7ec8e3;
  margin-bottom: 1.5rem;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.analysis-card {
  background: #0f0f1e;
  padding: 1.5rem;
  border-radius: 10px;
  border: 1px solid #16213e;
}

.analysis-card h3 {
  color: #7ec8e3;
  font-size: 0.9rem;
  margin-bottom: 1rem;
  text-transform: uppercase;
}

.strategy,
.risk-level,
.sentiment,
.action {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
}

.action.buy {
  color: #4fbdba;
}

.action.sell {
  color: #e94560;
}

.action.hold {
  color: #f39c12;
}

/* Portfolio Page */
.portfolio-page {
  padding: 2rem;
}

.portfolio-page h1 {
  color: #4fbdba;
  margin-bottom: 2rem;
}

.portfolio-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.summary-card {
  background: #1a1a2e;
  padding: 1.5rem;
  border-radius: 10px;
  border: 1px solid #16213e;
}

.summary-card h3 {
  color: #7ec8e3;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
}

.summary-card p {
  font-size: 2rem;
  font-weight: 700;
  color: white;
}

.positions-section,
.transactions-section {
  background: #1a1a2e;
  padding: 2rem;
  border-radius: 10px;
  border: 1px solid #16213e;
  margin-bottom: 2rem;
}

.positions-section h2,
.transactions-section h2 {
  color: #4fbdba;
  margin-bottom: 1.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
  .nav-links {
    display: none;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .charts-section {
    grid-template-columns: 1fr;
  }
  
  .trading-panel {
    grid-template-columns: 1fr;
  }
  
  .analysis-grid {
    grid-template-columns: 1fr;
  }
}
