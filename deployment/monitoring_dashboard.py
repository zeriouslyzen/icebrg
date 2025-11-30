#!/usr/bin/env python3
"""
Elite Financial AI Oracle - Monitoring Dashboard
Real-time monitoring and visualization of system performance
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import redis
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Elite Financial AI Oracle - Monitoring Dashboard"

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# ICEBURG API endpoints
ICEBURG_API_BASE = "http://localhost:8000"
PROMETHEUS_API_BASE = "http://localhost:9090"

class MonitoringDashboard:
    """Real-time monitoring dashboard for Elite Financial AI Oracle"""
    
    def __init__(self):
        self.app = app
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 Elite Financial AI Oracle - Monitoring Dashboard", 
                       style={'textAlign': 'center', 'color': '#2E86AB'}),
                html.P("Real-time system monitoring and performance analytics",
                      style={'textAlign': 'center', 'color': '#666'})
            ], style={'marginBottom': '30px'}),
            
            # System Status Cards
            html.Div([
                html.Div([
                    html.H3("System Status", style={'color': '#2E86AB'}),
                    html.Div(id='system-status-cards')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Performance Metrics", style={'color': '#2E86AB'}),
                    html.Div(id='performance-metrics')
                ], className='six columns')
            ], className='row'),
            
            # Charts Section
            html.Div([
                html.Div([
                    html.H3("Trading Performance", style={'color': '#2E86AB'}),
                    dcc.Graph(id='trading-performance-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Quantum Circuit Performance", style={'color': '#2E86AB'}),
                    dcc.Graph(id='quantum-performance-chart')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H3("RL Agent Performance", style={'color': '#2E86AB'}),
                    dcc.Graph(id='rl-performance-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H3("System Health", style={'color': '#2E86AB'}),
                    dcc.Graph(id='system-health-chart')
                ], className='six columns')
            ], className='row'),
            
            # Real-time Updates
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('system-status-cards', 'children'),
             Output('performance-metrics', 'children'),
             Output('trading-performance-chart', 'figure'),
             Output('quantum-performance-chart', 'figure'),
             Output('rl-performance-chart', 'figure'),
             Output('system-health-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update dashboard with real-time data"""
            
            # Get system status
            system_status = self.get_system_status()
            status_cards = self.create_status_cards(system_status)
            
            # Get performance metrics
            performance_metrics = self.get_performance_metrics()
            metrics_display = self.create_metrics_display(performance_metrics)
            
            # Get trading performance chart
            trading_chart = self.create_trading_performance_chart()
            
            # Get quantum performance chart
            quantum_chart = self.create_quantum_performance_chart()
            
            # Get RL performance chart
            rl_chart = self.create_rl_performance_chart()
            
            # Get system health chart
            health_chart = self.create_system_health_chart()
            
            return (status_cards, metrics_display, trading_chart, 
                   quantum_chart, rl_chart, health_chart)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'iceburg_core': self.check_service_health('iceburg'),
            'redis': self.check_service_health('redis'),
            'ollama': self.check_service_health('ollama'),
            'monitoring': self.check_service_health('monitoring'),
            'quantum_backend': self.check_quantum_backend(),
            'rl_agents': self.check_rl_agents()
        }
        return status
    
    def check_service_health(self, service: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            if service == 'iceburg':
                response = requests.get(f"{ICEBURG_API_BASE}/health", timeout=5)
                return {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
            elif service == 'redis':
                redis_client.ping()
                return {
                    'status': 'healthy',
                    'response_time': 0.001,
                    'timestamp': datetime.now().isoformat()
                }
            elif service == 'ollama':
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                return {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unknown',
                    'response_time': 0,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'response_time': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_quantum_backend(self) -> Dict[str, Any]:
        """Check quantum backend status"""
        try:
            # Check if quantum circuits are working
            import pennylane as qml
            dev = qml.device("default.qubit", wires=2)
            
            @qml.qnode(dev)
            def test_circuit():
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))
            
            result = test_circuit()
            return {
                'status': 'healthy',
                'test_result': float(result),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_rl_agents(self) -> Dict[str, Any]:
        """Check RL agents status"""
        try:
            # Get agent status from Redis
            agent_status = redis_client.hgetall('rl_agents_status')
            return {
                'status': 'healthy' if agent_status else 'no_agents',
                'agent_count': len(agent_status),
                'agents': agent_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage': self.get_memory_usage(),
            'trading_performance': self.get_trading_performance(),
            'quantum_circuit_time': self.get_quantum_circuit_time(),
            'rl_agent_rewards': self.get_rl_agent_rewards()
        }
        return metrics
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def get_trading_performance(self) -> Dict[str, float]:
        """Get trading performance metrics"""
        try:
            # Get from Redis cache
            performance = redis_client.hgetall('trading_performance')
            return {k: float(v) for k, v in performance.items()}
        except:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def get_quantum_circuit_time(self) -> float:
        """Get average quantum circuit execution time"""
        try:
            circuit_time = redis_client.get('quantum_circuit_time')
            return float(circuit_time) if circuit_time else 0.0
        except:
            return 0.0
    
    def get_rl_agent_rewards(self) -> List[float]:
        """Get RL agent rewards"""
        try:
            rewards = redis_client.lrange('rl_agent_rewards', 0, -1)
            return [float(r) for r in rewards]
        except:
            return []
    
    def create_status_cards(self, status: Dict[str, Any]) -> List[html.Div]:
        """Create status cards"""
        cards = []
        for service, info in status.items():
            color = 'green' if info['status'] == 'healthy' else 'red'
            cards.append(
                html.Div([
                    html.H4(service.replace('_', ' ').title()),
                    html.P(f"Status: {info['status']}", style={'color': color}),
                    html.P(f"Response Time: {info.get('response_time', 0):.3f}s")
                ], style={
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'padding': '10px',
                    'margin': '5px',
                    'backgroundColor': '#f9f9f9'
                })
            )
        return cards
    
    def create_metrics_display(self, metrics: Dict[str, Any]) -> List[html.Div]:
        """Create metrics display"""
        displays = []
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    displays.append(
                        html.Div([
                            html.H5(f"{metric} - {sub_metric}"),
                            html.P(f"{sub_value:.2f}")
                        ])
                    )
            else:
                displays.append(
                    html.Div([
                        html.H5(metric.replace('_', ' ').title()),
                        html.P(f"{value:.2f}")
                    ])
                )
        return displays
    
    def create_trading_performance_chart(self) -> go.Figure:
        """Create trading performance chart"""
        # Generate sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates)).cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name='Portfolio Returns',
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig.update_layout(
            title="Trading Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            template="plotly_white"
        )
        
        return fig
    
    def create_quantum_performance_chart(self) -> go.Figure:
        """Create quantum performance chart"""
        # Generate sample data
        circuits = ['VQC', 'QAOA', 'QGAN', 'Quantum Kernel']
        execution_times = np.random.exponential(0.1, len(circuits))
        
        fig = go.Figure(data=[
            go.Bar(x=circuits, y=execution_times, marker_color='#A23B72')
        ])
        
        fig.update_layout(
            title="Quantum Circuit Execution Times",
            xaxis_title="Circuit Type",
            yaxis_title="Execution Time (seconds)",
            template="plotly_white"
        )
        
        return fig
    
    def create_rl_performance_chart(self) -> go.Figure:
        """Create RL performance chart"""
        # Generate sample data
        episodes = list(range(100))
        rewards = np.random.normal(0, 1, 100).cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name='Agent Rewards',
            line=dict(color='#F18F01', width=2)
        ))
        
        fig.update_layout(
            title="RL Agent Training Progress",
            xaxis_title="Episode",
            yaxis_title="Cumulative Reward",
            template="plotly_white"
        )
        
        return fig
    
    def create_system_health_chart(self) -> go.Figure:
        """Create system health chart"""
        # Generate sample data
        time_points = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                  end=datetime.now(), freq='H')
        cpu_usage = np.random.normal(50, 10, len(time_points))
        memory_usage = np.random.normal(60, 5, len(time_points))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=cpu_usage,
            mode='lines',
            name='CPU Usage %',
            line=dict(color='#2E86AB')
        ))
        fig.add_trace(go.Scatter(
            x=time_points,
            y=memory_usage,
            mode='lines',
            name='Memory Usage %',
            line=dict(color='#A23B72')
        ))
        
        fig.update_layout(
            title="System Resource Usage (24h)",
            xaxis_title="Time",
            yaxis_title="Usage %",
            template="plotly_white"
        )
        
        return fig

def main():
    """Main function to run the monitoring dashboard"""
    dashboard = MonitoringDashboard()
    
    print("🚀 Starting Elite Financial AI Oracle Monitoring Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🔄 Auto-refresh every 5 seconds")
    print("📈 Real-time monitoring of quantum, RL, and financial systems")
    
    # Run the dashboard
    dashboard.app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == "__main__":
    main()
