# ICEBURG Design Tokens & Aesthetic Principles (KI)

This Knowledge Item documents the "Praxis" Design System used in the ICEBURG frontend.

## Core Aesthetic: Futuristic Noir
ICEBURG utilizes a high-contrast Black/White foundation, emphasizing depth through shadows and glows rather than color. 

### Design Principles
1.  **True Inversion**: Dark mode and Light mode are perfect inverses. Dark mode uses pure black `#000000` backgrounds with white text; Light mode uses off-white `#f8f8f6` backgrounds with black text.
2.  **Depth via Luminosity**: Elevation is represented by subtle background shifts (e.g., `#0a0a0a` vs `#111111`) and high-luminosity shadows.
3.  **Bio-Mimetic Motion**: UI elements pulse and shift based on "Brainwave" patterns (Alpha, Beta, Theta, Delta, Gamma).
4.  **Glassmorphism (Dark Mode Only)**: Subdued blur effects on headers and panels to create a sense of layered intelligence.

## Design Tokens (CSS Variables)

### Base Palette (Dark Mode)
| Variable | Value | Usage |
| :--- | :--- | :--- |
| `--bg-primary` | `#000000` | Main application background |
| `--bg-secondary`| `#0a0a0a` | Secondary surfaces / sidebar |
| `--bg-elevated` | `#1a1a1a` | Modals and floating panels |
| `--text-primary`| `#ffffff` | Primary readable content |
| `--text-tertiary`| `#888888` | Metadata and de-emphasized text |
| `--accent-primary`| `#ffffff` | Interactive highlights |

### Base Palette (Light Mode - Inversion)
| Variable | Value | Usage |
| :--- | :--- | :--- |
| `--bg-primary` | `#f8f8f6` | Soft off-white main background |
| `--bg-secondary`| `#f3f3f1` | Secondary surfaces |
| `--text-primary`| `#000000` | Primary readable content |

### Motion & Transitions
| Variable | Value |
| :--- | :--- |
| `--transition-base` | `300ms cubic-bezier(0.4, 0, 0.2, 1)` |
| `--transition-slow` | `500ms` |
| `--morph-duration` | `800ms` |

## Brainwave Animations
ICEBURG uses specific `@keyframes` to communicate system state and cognitive load:

- **Alpha Wave (8-12 Hz)**: Relaxed focus. Used for idle states. Yellow/Green gradients.
- **Theta Wave (4-8 Hz)**: Deep relaxation/Creativity. Used for complex synthesis. Purple/Violet.
- **Beta Wave (12-30 Hz)**: Alert focus. Used for active research. Orange/Red.
- **Delta Wave (0.1-4 Hz)**: Deep sleep/Rest. Deep Violet/Indigo.
- **Gamma Wave (30+ Hz)**: Peak performance. Gold/White. Used for breakthrough discoveries.

## Typography
- **Primary Font**: San Francisco (SF Pro), System Sans-Serif.
- **Readability**: Light mode uses a heavier weight (`450`) and extra letter spacing (`0.01em`) to optimize black-on-white legibility.
