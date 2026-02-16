# Browser Deceptive / Safe Browsing Audit

Audit date: 2025-02. Purpose: identify anything in the frontend that could trigger browser "Deceptive site ahead" or Safe Browsing warnings.

## Summary: No deceptive patterns found

The codebase does not implement behavior that typically triggers deceptive-site or malware warnings. Details below.

---

## Checked items

### 1. Credential / phishing-like prompts
- **Status: Clean.** No `prompt()` or modal asking for "Identity Verification", password, or verification code.
- The previous paywall flow (OBSIDIAN code) was fully disabled; `auth-check.js` is a no-op and the paywall handler in `main.js` no longer shows a modal or prompts for input.
- Remaining `prompt()` usage: `admin.js` and `js/testing.js` use `prompt('Template name:')` for template naming only. Benign.

### 2. Redirects
- **Status: Clean.** No redirect to external or unknown URLs.
- `auth-check.js` no longer redirects (returns immediately).
- All `window.location.href` / `location.replace` in the frontend are same-origin (e.g. `/app`, `/research`, `/pegasus`). No off-site redirects.

### 3. eval() / dynamic code execution
- **Status: Clean.** No `eval()` or `new Function()` in frontend JS. Reduces risk of heuristic or malware-style flagging.

### 4. External fetch / data exfiltration
- **Status: Clean.** All `fetch()` and `window.open()` targets are same-origin (e.g. `/api/...`, `${API_BASE}/api/...`). No sending of form data or credentials to third-party domains.

### 5. Password and API key handling
- **Status: Benign.** Password-type inputs are used only for:
  - Profile display (readonly placeholder).
  - Optional API keys (e.g. Gemini) stored in `localStorage` with `btoa` obfuscation; keys stay in the browser and are not sent to external servers. Comment in app: "Keys are stored locally in your browser."
- No credential harvesting or submission to third parties.

### 6. External resources (scripts, styles, embeds)
- **Status: Normal.** External URLs are limited to:
  - Trusted CDNs: Google Fonts, jsDelivr, d3js.org, cdnjs (highlight.js, Chart.js, Plotly, marked, etc.), ESM.run for web-llm.
  - TradingView widget embed in admin (known product).
  - DOI links (https://doi.org/...) in encyclopedia.
- No unknown or suspicious third-party scripts.

### 7. Hidden iframes / disguised forms
- **Status: Clean.** No iframes. No form `action` to external URLs.

### 8. Fake system / browser UI
- **Status: Clean.** No alerts or modals that mimic OS/browser updates, virus scans, or "verify your account" flows. Remaining `alert()` / `confirm()` are for app feedback (errors, confirmations). `window.testICEBURG` shows a test alert only when explicitly invoked (e.g. from console).

### 9. Obfuscation
- **Status: Benign.** Only `btoa`/`atob` for optional API key storage in the app; no heavy or suspicious obfuscation.

---

## Recommendations

1. **Keep paywall disabled as-is.** The current no-op `auth-check.js` and simplified paywall handler avoid any "verification" or code-entry flow that could be misread as phishing.
2. **innerHTML usage:** Many templates use `innerHTML` with dynamic content. If any of that content comes from untrusted APIs, sanitize (e.g. DOMPurify) to prevent XSS. This is a general security practice, not specific to deceptive-site warnings.
3. **Domain / hosting:** If 1cebrg.com was previously flagged, the cause was likely domain reputation or name similarity (e.g. to "ICEBRG" security brand), not in-app behavior. Keeping the app free of deceptive patterns supports a successful Safe Browsing review.

---

## Files reviewed

- `frontend/public/auth-check.js` – no-op; no redirect.
- `frontend/main.js` – paywall block simplified; no credential prompt.
- `frontend/app.html` – sidebar links; password/API key inputs (local only).
- All `frontend/**/*.js` and `frontend/**/*.html` for prompt/redirect/fetch/eval/iframe/form and external URLs.
