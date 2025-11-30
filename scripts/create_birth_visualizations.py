#!/usr/bin/env python3
"""
Create aesthetic visualizations for birth analysis results.
Includes charts, animations, and interactive displays.
"""

import sys
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle, Rectangle
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available, creating HTML visualizations instead")

def create_celestial_chart(data: Dict[str, Any], output_dir: Path):
    """Create celestial positions chart"""
    
    if not HAS_MATPLOTLIB:
        return create_html_celestial_chart(data, output_dir)
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Get celestial positions
    positions = data.get("molecular_imprint", {}).get("celestial_positions", {})
    
    # Create polar plot for celestial positions
    ax = plt.subplot(111, projection='polar')
    ax.set_facecolor('#0a0a0a')
    
    # Plot each celestial body
    colors = {
        'sun': '#FFD700',
        'moon': '#C0C0C0',
        'mars': '#FF6B6B',
        'mercury': '#87CEEB',
        'jupiter': '#FFA500',
        'venus': '#FFB6C1',
        'saturn': '#F0E68C'
    }
    
    for body, pos_data in positions.items():
        if isinstance(pos_data, dict):
            ra = math.radians(pos_data.get('ra', 0))
            dec = pos_data.get('dec', 0)
            distance = pos_data.get('distance', 1.0)
            
            # Normalize distance for visualization (0.3 to 1.0 AU scale)
            radius = 0.3 + (distance / 10.0) * 0.7
            
            color = colors.get(body, '#FFFFFF')
            ax.scatter(ra, radius, s=300, c=color, alpha=0.8, label=body.upper(), edgecolors='white', linewidths=2)
            ax.text(ra, radius, body.upper(), color='white', fontsize=8, ha='center', va='center')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1.2)
    ax.set_title('Celestial Positions at Birth', color='white', fontsize=16, pad=20)
    ax.grid(True, color='#333333', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "celestial_positions.png"
    plt.savefig(output_file, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    print(f"✅ Created: {output_file}")
    plt.close()

def create_behavioral_traits_chart(data: Dict[str, Any], output_dir: Path):
    """Create behavioral traits radar chart"""
    
    if not HAS_MATPLOTLIB:
        return create_html_behavioral_chart(data, output_dir)
    
    traits = data.get("behavioral_predictions", {})
    if not traits:
        return
    
    # Prepare data
    categories = list(traits.keys())
    values = list(traits.values())
    
    # Normalize values to 0-1 range for radar chart
    max_val = max(values) if values else 1.0
    normalized_values = [v / max_val if max_val > 0 else 0 for v in values]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Calculate angles
    angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
    angles += angles[:1]  # Complete the circle
    normalized_values += normalized_values[:1]
    
    # Plot
    ax.plot(angles, normalized_values, 'o-', linewidth=2, color='#00D4FF', alpha=0.8)
    ax.fill(angles, normalized_values, alpha=0.25, color='#00D4FF')
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], color='white', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='white', fontsize=8)
    ax.grid(True, color='#333333', alpha=0.3)
    
    plt.title('Behavioral Traits Profile', color='white', fontsize=16, pad=20)
    
    plt.tight_layout()
    output_file = output_dir / "behavioral_traits.png"
    plt.savefig(output_file, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    print(f"✅ Created: {output_file}")
    plt.close()

def create_tcm_organs_chart(data: Dict[str, Any], output_dir: Path):
    """Create TCM organ system chart"""
    
    if not HAS_MATPLOTLIB:
        return create_html_tcm_chart(data, output_dir)
    
    tcm_data = data.get("tcm_predictions", {})
    if not tcm_data:
        return
    
    # Extract data
    planets = []
    organs = []
    strengths = []
    elements = []
    
    for planet, info in tcm_data.items():
        planets.append(planet.upper())
        organs.append(info.get('organ', 'Unknown'))
        strengths.append(info.get('strength', 0))
        elements.append(info.get('element', 'Unknown'))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Color map for elements
    element_colors = {
        'fire': '#FF4444',
        'water': '#4444FF',
        'earth': '#8B4513',
        'metal': '#C0C0C0',
        'wood': '#228B22'
    }
    
    colors = [element_colors.get(elem.lower(), '#FFFFFF') for elem in elements]
    
    bars = ax.barh(planets, strengths, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add organ labels
    for i, (planet, organ, strength) in enumerate(zip(planets, organs, strengths)):
        ax.text(strength + 0.01, i, f"{organ.title()}", 
                va='center', color='white', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Organ Strength', color='white', fontsize=12)
    ax.set_ylabel('Celestial Body', color='white', fontsize=12)
    ax.set_title('TCM Organ System Correlations', color='white', fontsize=16, pad=20)
    ax.set_xlim(0, max(strengths) * 1.2 if strengths else 1.0)
    ax.tick_params(colors='white')
    ax.grid(True, axis='x', color='#333333', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "tcm_organs.png"
    plt.savefig(output_file, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    print(f"✅ Created: {output_file}")
    plt.close()

def create_animated_timeline(data: Dict[str, Any], output_dir: Path):
    """Create animated timeline showing celestial influences over time"""
    
    if not HAS_MATPLOTLIB:
        return
    
    birth_date = datetime.fromisoformat(data.get("birth_data", {}).get("date", ""))
    current_date = datetime.fromisoformat(data.get("analysis_date", ""))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Time range
    years = np.arange(1991, 2026)
    
    # Simulate trait variations over time (simplified)
    traits = data.get("behavioral_predictions", {})
    trait_names = list(traits.keys())[:4]  # Limit to 4 for clarity
    
    lines = []
    for trait in trait_names:
        base_value = traits.get(trait, 0.5)
        # Add some variation over time
        values = base_value + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(years)))
        line, = ax.plot(years, values, label=trait.replace('_', ' ').title(), linewidth=2, alpha=0.8)
        lines.append(line)
    
    ax.set_xlabel('Year', color='white', fontsize=12)
    ax.set_ylabel('Trait Amplification', color='white', fontsize=12)
    ax.set_title('Behavioral Traits Over Time', color='white', fontsize=16, pad=20)
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='#333333', alpha=0.3)
    ax.set_xlim(1991, 2025)
    
    # Mark birth date and current date
    ax.axvline(x=1991, color='#00D4FF', linestyle='--', linewidth=2, alpha=0.7, label='Birth')
    ax.axvline(x=2025, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.7, label='Today')
    
    def animate(frame):
        # Update to show progression
        current_year = 1991 + (frame / 10) * (2025 - 1991)
        ax.set_xlim(1991, max(1991 + (frame / 10) * (2025 - 1991), 1992))
        return lines
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True, repeat=True)
    
    output_file = output_dir / "animated_timeline.gif"
    anim.save(output_file, writer='pillow', fps=20)
    print(f"✅ Created: {output_file}")
    plt.close()

def create_html_visualization(data: Dict[str, Any], output_dir: Path):
    """Create interactive HTML visualization"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Birth Analysis Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00D4FF;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .card h2 {{
            color: #00D4FF;
            margin-top: 0;
        }}
        canvas {{
            max-height: 400px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌌 Birth Analysis Visualization</h1>
        
        <div class="card">
            <h2>Birth Information</h2>
            <p><strong>Date:</strong> {data.get('birth_data', {}).get('date', 'N/A')}</p>
            <p><strong>Location:</strong> Parkland Hospital, Dallas, TX</p>
            <p><strong>Age:</strong> {data.get('birth_data', {}).get('age_years', 0):.1f} years</p>
            <p><strong>Analysis Date:</strong> {data.get('analysis_date', 'N/A')}</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Behavioral Traits</h2>
                <canvas id="traitsChart"></canvas>
            </div>
            
            <div class="card">
                <h2>TCM Organ Systems</h2>
                <canvas id="tcmChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Behavioral Traits Chart
        const traitsData = {json.dumps(data.get('behavioral_predictions', {}))};
        const traitsCtx = document.getElementById('traitsChart').getContext('2d');
        new Chart(traitsCtx, {{
            type: 'radar',
            data: {{
                labels: Object.keys(traitsData).map(k => k.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                datasets: [{{
                    label: 'Trait Amplification',
                    data: Object.values(traitsData),
                    backgroundColor: 'rgba(0, 212, 255, 0.2)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    pointBackgroundColor: 'rgba(0, 212, 255, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(0, 212, 255, 1)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            color: '#ffffff'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }},
                        pointLabels: {{
                            color: '#ffffff'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#ffffff'
                        }}
                    }}
                }}
            }}
        }});
        
        // TCM Chart
        const tcmData = {json.dumps(data.get('tcm_predictions', {}))};
        const tcmCtx = document.getElementById('tcmChart').getContext('2d');
        const tcmLabels = Object.keys(tcmData).map(k => k.toUpperCase());
        const tcmStrengths = Object.values(tcmData).map(v => v.strength || 0);
        
        new Chart(tcmCtx, {{
            type: 'bar',
            data: {{
                labels: tcmLabels,
                datasets: [{{
                    label: 'Organ Strength',
                    data: tcmStrengths,
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            color: '#ffffff'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            color: '#ffffff'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#ffffff'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    output_file = output_dir / "visualization.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Created: {output_file}")

def create_html_celestial_chart(data: Dict[str, Any], output_dir: Path):
    """Create HTML celestial positions chart"""
    # Placeholder - would create SVG or use canvas
    pass

def create_html_behavioral_chart(data: Dict[str, Any], output_dir: Path):
    """Create HTML behavioral chart"""
    # Already in main HTML
    pass

def create_html_tcm_chart(data: Dict[str, Any], output_dir: Path):
    """Create HTML TCM chart"""
    # Already in main HTML
    pass

def main():
    """Main function to create all visualizations"""
    
    # Load data
    data_file = Path("data/birth_analysis_results.json")
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Run test_birth_analysis.py first!")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create output directory
    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Create visualizations
    if HAS_MATPLOTLIB:
        print("📊 Creating matplotlib visualizations...")
        create_celestial_chart(data, output_dir)
        create_behavioral_traits_chart(data, output_dir)
        create_tcm_organs_chart(data, output_dir)
        try:
            create_animated_timeline(data, output_dir)
        except Exception as e:
            print(f"⚠️  Could not create animation: {e}")
    else:
        print("📊 Creating HTML visualizations...")
    
    # Always create HTML visualization
    create_html_visualization(data, output_dir)
    
    print()
    print("=" * 80)
    print("✅ VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"📁 Output directory: {output_dir}")
    print(f"🌐 Open visualization.html in a browser to view interactive charts")

if __name__ == "__main__":
    main()

