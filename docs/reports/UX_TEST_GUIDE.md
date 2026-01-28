# ICEBURG Fast Mode Secretary - UX Test Guide

## Quick Start

1. **Open the UI**: http://localhost:3000
2. **Select Mode**: Choose "Fast" or "Chat" mode from the mode selector
3. **Select Agent**: Ensure "Secretary" is selected (or leave as "auto" - it will default to Secretary in chat mode)
4. **Start Testing**: Use the conversation interface to test

## Test Questions for Startup Contest

### Basic Identity & Capabilities
1. "Who are you and what is ICEBURG?"
2. "What are ICEBURG's main capabilities?"
3. "Tell me about the different agents in ICEBURG. What does each one do?"

### Self-Research & Architecture
4. "Can you research yourself? What can you tell me about ICEBURG's architecture?"
5. "What tools and abilities do you have access to?"
6. "Can you use those tools to find more information about ICEBURG's research capabilities?"

### Differentiation & Usage
7. "What makes ICEBURG different from other AI systems?"
8. "How would someone use ICEBURG for a research project?"

## What to Look For

### ✅ Success Indicators
- Responses mention ICEBURG and its capabilities
- Mentions agents (Surveyor, Dissident, Synthesist, Oracle, Secretary)
- Logical conversation flow
- Follow-up questions are understood in context
- Response times: 20-40 seconds (acceptable for local Ollama)
- Streaming works smoothly (character-by-character)

### ⚠️ Issues to Note
- Response times > 60 seconds
- Generic or off-topic responses
- No mention of ICEBURG when asked about it
- Conversation continuity breaks
- WebSocket connection failures

## Testing Checklist

- [ ] UI loads correctly at http://localhost:3000
- [ ] Mode selector shows "Fast" or "Chat" option
- [ ] Can send queries and receive responses
- [ ] Streaming works (text appears character-by-character)
- [ ] Responses are relevant and mention ICEBURG
- [ ] Conversation history persists
- [ ] Follow-up questions work
- [ ] Mobile view works (if testing on mobile)

## Expected Behavior

**Fast/Chat Mode:**
- Uses Secretary agent
- Fast responses (20-40s for local Ollama)
- Knowledgeable about ICEBURG
- Can explain capabilities, agents, architecture
- Maintains conversation context

**Response Quality:**
- Mentions ICEBURG when relevant
- Explains agent roles clearly
- Describes capabilities accurately
- Provides practical usage guidance

## Troubleshooting

**If responses are slow:**
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Check model is loaded: `ollama list`
- Consider using smaller model in settings

**If responses are generic:**
- Ensure mode is set to "Chat" or "Fast"
- Check agent is set to "Secretary" or "auto"
- Verify Ollama is using correct model

**If WebSocket fails:**
- System will automatically fallback to HTTP/SSE
- Check browser console for errors
- Try refreshing the page

## Notes for Startup Contest Demo

- **Demo Flow**: Start with identity question, then capabilities, then architecture
- **Highlight**: Multi-agent coordination, local-first operation, research capabilities
- **Show**: Streaming responses, conversation continuity, knowledge of ICEBURG
- **Emphasize**: Self-awareness (can research itself), tool usage, practical applications

