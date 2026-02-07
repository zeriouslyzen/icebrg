(function() {
    // Bypass for E2E/UX testing: localhost or ?test=1 skips redirect
    var isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    if (isLocal || window.location.search.indexOf('test=1') !== -1) return;
    // Check for clearance token
    const clearance = localStorage.getItem('iceburg_clearance');
    
    // If no clearance, redirect to app with locked error
    if (!clearance) {
        // Save current URL to redirect back after login
        sessionStorage.setItem('redirect_after_login', window.location.pathname);
        
        // Immediate redirect
        window.location.replace('/app?error=locked&from=' + encodeURIComponent(window.location.pathname));
        
        // Stop further execution
        throw new Error('Access Denied: Clearance required');
    }
})();
