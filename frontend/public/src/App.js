import React, { useState, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useNavigate
} from 'react-router-dom';
import axios from 'axios';
import io from 'socket.io-client';
import './App.css';

// Configure axios defaults
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
axios.defaults.baseURL = API_URL;

// Components
import Dashboard from './components/Dashboard';
import TradingInterface from './components/TradingInterface';
import Portfolio from './components/Portfolio';
import MarketAnalysis from './components/MarketAnalysis';
import Login from './components/Login';
import Register from './components/Register';
import Navbar from './components/Navbar';

// Auth context
const AuthContext = React.createContext();

function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [socket, setSocket] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUserProfile();
      initializeSocket();
    } else {
      setLoading(false);
    }

    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, [token]);

  const fetchUserProfile = async () => {
    try {
      const response = await axios.get('/api/auth/profile');
      setUser(response.data);
    } catch (error) {
      console.error('Failed to fetch user profile:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const initializeSocket = () => {
    const newSocket = io(API_URL, {
      auth: {
        token: token
      }
    });

    newSocket.on('connect', () => {
      console.log('Connected to trading server');
    });

    newSocket.on('trade_executed', (data) => {
      console.log('Trade executed:', data);
      // Handle trade execution updates
    });

    newSocket.on('market_update', (data) => {
      console.log('Market update:', data);
      // Handle real-time market updates
    });

    setSocket(newSocket);
  };

  const login = async (username, password) => {
    try {
      const response = await axios.post('/api/auth/login', {
        username,
        password
      });

      const { access_token, user } = response.data;
      
      localStorage.setItem('token', access_token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      setToken(access_token);
      setUser(user);
      initializeSocket();
      
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'Login failed'
      };
    }
  };

  const register = async (username, email, password) => {
    try {
      const response = await axios.post('/api/auth/register', {
        username,
        email,
        password
      });

      const { access_token, user } = response.data;
      
      localStorage.setItem('token', access_token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      setToken(access_token);
      setUser(user);
      initializeSocket();
      
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'Registration failed'
      };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
    
    if (socket) {
      socket.disconnect();
      setSocket(null);
    }
    
    setToken(null);
    setUser(null);
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading Fintech Trading System...</p>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={{ user, token, socket, login, register, logout }}>
      <Router>
        <div className="app">
          {user && <Navbar user={user} onLogout={logout} />}
          
          <Routes>
            <Route 
              path="/login" 
              element={
                user ? <Navigate to="/dashboard" /> : <Login />
              } 
            />
            
            <Route 
              path="/register" 
              element={
                user ? <Navigate to="/dashboard" /> : <Register />
              } 
            />
            
            <Route 
              path="/dashboard" 
              element={
                user ? <Dashboard /> : <Navigate to="/login" />
              } 
            />
            
            <Route 
              path="/trading" 
              element={
                user ? <TradingInterface socket={socket} /> : <Navigate to="/login" />
              } 
            />
            
            <Route 
              path="/portfolio" 
              element={
                user ? <Portfolio /> : <Navigate to="/login" />
              } 
            />
            
            <Route 
              path="/analysis" 
              element={
                user ? <MarketAnalysis /> : <Navigate to="/login" />
              } 
            />
            
            <Route 
              path="/" 
              element={
                user ? <Navigate to="/dashboard" /> : <Navigate to="/login" />
              } 
            />
          </Routes>
        </div>
      </Router>
    </AuthContext.Provider>
  );
}

export default App;
export { AuthContext };
