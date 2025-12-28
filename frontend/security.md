# ICEBURG Frontend Security Guide

## Security Features Implemented

### 1. Input Validation
- All user inputs are validated before sending to API
- Query length limits (max 10,000 characters)
- File upload validation (type, size, extension)
- Pattern detection for suspicious content

### 2. XSS Protection
- HTML escaping for all user-generated content
- Content Security Policy (CSP) headers
- Markdown sanitization
- No inline scripts or unsafe eval

### 3. CSRF Protection
- Same-origin policy enforcement
- Secure cookie handling
- Origin validation for WebSocket connections

### 4. Secure Connections
- HTTPS/WSS in production
- Secure WebSocket connections
- Environment-based configuration

### 5. File Upload Security
- File type validation
- File size limits (10MB max)
- Extension whitelist
- Path traversal prevention
- Content type verification

### 6. API Security
- Rate limiting (60 requests/minute)
- Input sanitization
- Error message sanitization
- Request validation

### 7. Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Referrer-Policy: strict-origin-when-cross-origin
- Content-Security-Policy
- HSTS (HTTPS only)

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Production (use HTTPS/WSS)
# VITE_API_URL=https://api.iceburg.ai
# VITE_WS_URL=wss://api.iceburg.ai

# Feature Flags
VITE_ENABLE_WEB_SEARCH=false
VITE_ENABLE_IMAGE_GENERATION=false
```

### Production Deployment

1. Use HTTPS for all connections
2. Set `ENVIRONMENT=production` on backend
3. Configure `ALLOWED_ORIGINS` on backend
4. Enable rate limiting
5. Use secure cookies
6. Enable HSTS
7. Regular security audits

## Best Practices

1. **Never trust user input** - Always validate and sanitize
2. **Use HTTPS in production** - Encrypt all communications
3. **Implement rate limiting** - Prevent abuse
4. **Keep dependencies updated** - Regular security patches
5. **Monitor for vulnerabilities** - Use security scanning tools
6. **Regular security audits** - Review code and dependencies

## Security Checklist

- [x] Input validation and sanitization
- [x] XSS protection
- [x] CSRF protection
- [x] Secure file uploads
- [x] Rate limiting
- [x] Security headers
- [x] HTTPS/WSS support
- [x] Environment-based configuration
- [x] Error message sanitization
- [x] Content Security Policy

