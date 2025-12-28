# Root Organization and UI Refactor Plan

This plan focuses on cleaning up the project root, updating project documentation with recent major research/optimizations, and disabling the "orb color system" (brainwave animations) in the UI.

## Proposed Changes

### [Component] Root Directory Organization
- [x] Move all `server_*.log` files to `logs/`
- [x] Move `backup_*.tar.gz` to `backups/`
- [x] Move `ConstituituinVibe.txt` to `docs/`
- [x] Move `submit_knowledge.py` to `scripts/`
- [x] Move `CHAT_MODE_SECRETARY_ONLY.md`, `CODEBASE_NOTICE.md`, `CURRENT_STATE.md` to `docs/` or specialized subfolders.

### [Component] Documentation Updates
- #### [MODIFY] [README.md](file:///Users/jackdanger/Desktop/Projects/iceburg/README.md)
  - Update architecture section with new research (Psyche, Linguistics, Protocol).
  - Add M4 optimization and Fine-tuning pipeline to features.
- #### [MODIFY] [CHANGELOG.md](file:///Users/jackdanger/Desktop/Projects/iceburg/CHANGELOG.md)
  - Add recent milestones: M4 Optimization, Fine-Tuning Manager, Deep Research Reports (Psyche, Protocol, Linguistics).

### [Component] Frontend UI
- #### [MODIFY] [styles.css](file:///Users/jackdanger/Desktop/Projects/iceburg/frontend/styles.css)
  - Disable the color-morphing animations for `.pulsing-ball-icon` (Alpha/Theta/Beta/Gamma colors).
  - Simplify orb styling to a static or neutral pulse (e.g., just white/black without shifting colors).

## Verification Plan

### Automated Tests
- Run `ls` in root to verify cleanup.
- Verify `npm run dev` (if applicable) doesn't break.

### Manual Verification
- Open `frontend/index.html` and verify the orbs (sidebar/settings icons) no longer shift colors based on brainwave cycles.
- Check `README.md` and `CHANGELOG.md` for accuracy.

## Phase 5: Verification & Debugging

### System Fixes Applied
- **Critical Fix**: Sanitized `PersistentMemoryAPI` metadata to prevent `NoneType` crashes in ChromaDB.
- **Performance**: Increased global research timeout to 300s.
- **Bug Fix**: Fixed `UnboundLocalError` in LLM fallback reporting.
- **Bug Fix**: Fixed `UnboundLocalError` (shadowed `Path`) in Surveyor agent.

### Verification Steps
1. **Memory Test**: Verify conversational continuity (name recall) in "Fast Chat".
2. **Research Test**: Verify "Deep Research" completes without timeout.
3. **Log Check**: Ensure no new tracebacks in `api_server.log`.

## Phase 5: Verification & Debugging

### System Fixes Applied
- **Critical Fix**: Sanitized `PersistentMemoryAPI` metadata to prevent `NoneType` crashes in ChromaDB.
- **Performance**: Increased global research timeout to 300s.
- **Bug Fix**: Fixed `UnboundLocalError` in LLM fallback reporting.

### Verification Steps
1. **Memory Test**: Verify conversational continuity (name recall) in "Fast Chat".
2. **Research Test**: Verify "Deep Research" completes without timeout.
3. **Log Check**: Ensure no new tracebacks in `api_server.log`.
