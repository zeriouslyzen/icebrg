# Visual Generation Examples

This directory contains example Visual TSL specifications for common UI patterns.

## Examples

### 1. Calculator (`calculator.json`)
A macOS-style calculator with:
- Number buttons (0-9)
- Operation buttons (+, -, ×, ÷)
- Special functions (C, ±, %)
- Display area
- Grid layout
- macOS design theme

**Usage:**
```bash
python -m src.iceburg.cli gen-ui "Create a macOS calculator" --backends all
```

### 2. Dashboard (`dashboard.json`)
An analytics dashboard with:
- Header with navigation
- Sidebar with user profile
- Metrics cards (users, revenue, orders, conversion)
- Chart placeholder
- Responsive grid layout
- Professional theme

**Usage:**
```bash
python -m src.iceburg.cli gen-ui "Create an analytics dashboard" --backends all
```

### 3. Todo App (`todo_app.json`)
A task management application with:
- Add new todo form
- Filter controls (All, Active, Completed)
- Todo list with checkboxes
- Delete functionality
- Item counter
- Clean, modern design

**Usage:**
```bash
python -m src.iceburg.cli gen-ui "Create a todo application" --backends all
```

### 4. Chat Interface (`chat_interface.json`)
A messaging interface with:
- Chat header with avatar and status
- Message history
- Input area with send button
- Quick action buttons
- Real-time messaging layout
- Chat-optimized theme

**Usage:**
```bash
python -m src.iceburg.cli gen-ui "Create a chat interface" --backends all
```

## Loading Examples

You can load these examples programmatically:

```python
from src.iceburg.iir.visual_tsl import UISpec

# Load calculator example
calculator_spec = UISpec.load_from_file("examples/visual_generation/calculator.json")

# Use with Visual Architect
from src.iceburg.agents.visual_architect import VisualArchitect
architect = VisualArchitect()

# Convert to IR and compile
visual_ir = architect._tsl_to_ir(calculator_spec, verbose=True)
artifacts = architect.backend_registry.compile_all(visual_ir)
```

## Customizing Examples

1. **Modify Components**: Edit the `components` array to add/remove UI elements
2. **Change Theme**: Update the `theme` section for different colors and styling
3. **Adjust Layout**: Modify the `layout` section for different arrangements
4. **Add Interactions**: Extend the `interactions` array for more functionality
5. **Update Contracts**: Modify `security_contracts` for different validation rules

## Best Practices

1. **Use Semantic IDs**: Give components meaningful, unique IDs
2. **Include Accessibility**: Always add ARIA labels and roles
3. **Set Performance Budgets**: Define realistic performance constraints
4. **Validate Contracts**: Include appropriate security and accessibility contracts
5. **Test Multiple Backends**: Verify your spec works across all target platforms

## Contributing

To add new examples:

1. Create a new `.json` file following the Visual TSL format
2. Include comprehensive component definitions
3. Add proper accessibility attributes
4. Set appropriate performance budgets
5. Include all necessary security contracts
6. Update this README with usage instructions

## File Structure

```
examples/visual_generation/
├── README.md              # This file
├── calculator.json        # macOS calculator example
├── dashboard.json         # Analytics dashboard example
├── todo_app.json          # Todo application example
└── chat_interface.json    # Chat interface example
```

Each example file contains:
- Complete Visual TSL specification
- All necessary components and interactions
- Accessibility requirements
- Performance budgets
- Security contracts
- Theme definitions
