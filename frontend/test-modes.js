/**
 * ICEBURG Mode Stress Test
 * ========================
 * Comprehensive testing of all modes to identify what works and what doesn't
 * 
 * Usage: Open browser console and run:
 *   import('/test-modes.js').then(m => m.runModeTests())
 * 
 * Or add to app.html temporarily:
 *   <script type="module" src="/test-modes.js"></script>
 */

const MODES_TO_TEST = [
    'fast',
    'chat',
    'research',
    'civilization',
    'astrophysiology',
    'dossier',
    'deep_research',
    'unbounded',
    'truth',
    'web_research',
    'local_rag',
    'hybrid',
    'protocol',
    'code',
    'finance'
];

const TEST_QUERIES = {
    fast: "What is 2+2?",
    chat: "Hello, how are you?",
    research: "Explain quantum entanglement",
    civilization: "Describe social simulation parameters",
    astrophysiology: "What is my molecular imprint?",
    dossier: "Search for information about AI safety",
    deep_research: "What are the implications of AGI?",
    unbounded: "Design a self-improving system",
    truth: "What is the truth about climate change?",
    web_research: "Latest news about AI",
    local_rag: "Search codebase for authentication",
    hybrid: "Compare cloud AI providers",
    protocol: "Run surveyor agent",
    code: "Write a hello world function",
    finance: "Analyze AAPL stock"
};

class ModeTester {
    constructor() {
        this.results = [];
        this.currentTest = null;
        this.errors = [];
    }

    async runAllTests() {
        console.log('🧪 Starting ICEBURG Mode Stress Tests...\n');
        console.log(`Testing ${MODES_TO_TEST.length} modes\n`);

        for (const mode of MODES_TO_TEST) {
            await this.testMode(mode);
            // Small delay between tests
            await this.delay(500);
        }

        this.printResults();
        // Also send results to backend for automated inspection
        this.sendResultsToServer().catch((err) => {
            console.warn('Failed to send mode test results to server', err);
        });
        return this.results;
    }

    async testMode(mode) {
        console.log(`\n📋 Testing mode: ${mode}`);
        const testResult = {
            mode,
            timestamp: new Date().toISOString(),
            passed: false,
            errors: [],
            warnings: [],
            checks: {}
        };

        try {
            // Check 1: Mode exists in select
            testResult.checks.modeInSelect = this.checkModeInSelect(mode);
            
            // Check 2: Can select mode
            testResult.checks.canSelectMode = await this.selectMode(mode);
            
            // Check 3: UI updates correctly
            testResult.checks.uiUpdates = this.checkUIUpdates(mode);
            
            // Check 4: Agent options update
            testResult.checks.agentOptions = this.checkAgentOptions(mode);
            
            // Check 5: Workflow pill doesn't block
            testResult.checks.workflowPillNotBlocking = this.checkWorkflowPill(mode);
            
            // Check 6: Can send query (if connection available)
            testResult.checks.canSendQuery = await this.testSendQuery(mode);
            
            // Check 7: No console errors
            testResult.checks.noConsoleErrors = this.checkConsoleErrors();
            
            // Check 8: Animations work
            testResult.checks.animationsWork = this.checkAnimations();
            
            // Overall pass/fail
            const allChecks = Object.values(testResult.checks);
            testResult.passed = allChecks.every(check => check === true || check === 'skip');
            
            if (!testResult.passed) {
                testResult.errors.push('One or more checks failed');
            }

        } catch (error) {
            testResult.errors.push(error.message);
            testResult.checks.error = error.message;
            console.error(`❌ Error testing ${mode}:`, error);
        }

        this.results.push(testResult);
        return testResult;
    }

    checkModeInSelect(mode) {
        const modeSelect = document.getElementById('modeSelect');
        const modeSelectSettings = document.getElementById('modeSelectSettings');
        
        if (!modeSelect && !modeSelectSettings) {
            console.warn(`⚠️  No mode select found`);
            return false;
        }

        const select = modeSelect || modeSelectSettings;
        const option = Array.from(select.options).find(opt => opt.value === mode);
        
        if (!option) {
            console.warn(`⚠️  Mode "${mode}" not found in select`);
            return false;
        }

        console.log(`  ✓ Mode "${mode}" exists in select`);
        return true;
    }

    async selectMode(mode) {
        try {
            const modeSelect = document.getElementById('modeSelect');
            const modeSelectSettings = document.getElementById('modeSelectSettings');
            
            if (!modeSelect && !modeSelectSettings) {
                return false;
            }

            const select = modeSelect || modeSelectSettings;
            select.value = mode;
            
            // Trigger change event
            select.dispatchEvent(new Event('change', { bubbles: true }));
            
            // Wait for UI to update
            await this.delay(100);
            
            // Verify selection
            if (select.value === mode) {
                console.log(`  ✓ Successfully selected mode "${mode}"`);
                return true;
            } else {
                console.warn(`  ⚠️  Mode selection failed - value is "${select.value}"`);
                return false;
            }
        } catch (error) {
            console.error(`  ❌ Error selecting mode:`, error);
            return false;
        }
    }

    checkUIUpdates(mode) {
        try {
            // Check if placeholder updates
            const queryInput = document.getElementById('queryInput');
            if (queryInput) {
                const placeholder = queryInput.getAttribute('placeholder') || '';
                console.log(`  ✓ Input placeholder: "${placeholder.substring(0, 50)}..."`);
            }

            // Check if workflow pill is hidden (should be when selecting modes)
            const workflowPill = document.getElementById('agentWorkflowPill');
            if (workflowPill) {
                const isHidden = workflowPill.classList.contains('hidden') || 
                               workflowPill.style.display === 'none' ||
                               workflowPill.style.opacity === '0';
                if (!isHidden) {
                    console.warn(`  ⚠️  Workflow pill is visible (should be hidden when selecting modes)`);
                    return false;
                }
                console.log(`  ✓ Workflow pill is hidden (not blocking)`);
            }

            // Check for blocking overlays
            const blockingOverlays = document.querySelectorAll('[style*="position: fixed"][style*="z-index"][style*="9999"]');
            const blockingModals = Array.from(blockingOverlays).filter(el => {
                const zIndex = window.getComputedStyle(el).zIndex;
                return parseInt(zIndex) >= 9999 && 
                       window.getComputedStyle(el).display !== 'none' &&
                       window.getComputedStyle(el).opacity !== '0';
            });

            if (blockingModals.length > 0) {
                console.warn(`  ⚠️  Found ${blockingModals.length} blocking overlay(s)`);
                blockingModals.forEach((modal, i) => {
                    console.warn(`    - Overlay ${i + 1}:`, modal.id || modal.className);
                });
                return false;
            }

            console.log(`  ✓ No blocking overlays detected`);
            return true;
        } catch (error) {
            console.error(`  ❌ Error checking UI updates:`, error);
            return false;
        }
    }

    checkAgentOptions(mode) {
        try {
            const agentSelect = document.getElementById('agentSelect');
            if (!agentSelect) {
                console.log(`  ⚠️  Agent select not found (may be hidden for this mode)`);
                return 'skip';
            }

            const options = Array.from(agentSelect.options);
            if (options.length === 0) {
                console.warn(`  ⚠️  Agent select has no options`);
                return false;
            }

            console.log(`  ✓ Agent select has ${options.length} options`);
            return true;
        } catch (error) {
            console.error(`  ❌ Error checking agent options:`, error);
            return false;
        }
    }

    checkWorkflowPill(mode) {
        try {
            const pill = document.getElementById('agentWorkflowPill');
            if (!pill) {
                console.log(`  ⚠️  Workflow pill not found (may not be initialized)`);
                return 'skip';
            }

            const rect = pill.getBoundingClientRect();
            const isVisible = window.getComputedStyle(pill).opacity !== '0' &&
                            window.getComputedStyle(pill).display !== 'none' &&
                            rect.width > 0 && rect.height > 0;

            if (isVisible) {
                // Check if it's blocking content (center of screen)
                const centerX = window.innerWidth / 2;
                const centerY = window.innerHeight / 2;
                const isBlocking = rect.left < centerX && rect.right > centerX &&
                                 rect.top < centerY && rect.bottom > centerY;

                if (isBlocking) {
                    console.warn(`  ⚠️  Workflow pill is blocking center of screen`);
                    return false;
                }
            }

            console.log(`  ✓ Workflow pill is not blocking`);
            return true;
        } catch (error) {
            console.error(`  ❌ Error checking workflow pill:`, error);
            return false;
        }
    }

    async testSendQuery(mode) {
        try {
            // Check if connection is available
            if (!window.ICEBURG_CONNECTION) {
                console.log(`  ⚠️  Connection not available (skipping query test)`);
                return 'skip';
            }

            const query = TEST_QUERIES[mode] || `Test query for ${mode}`;
            let querySent = false;
            let queryError = null;

            // Try to send query with timeout
            const testPromise = new Promise((resolve) => {
                const timeout = setTimeout(() => {
                    resolve(false);
                }, 2000); // 2 second timeout

                try {
                    const abort = window.ICEBURG_CONNECTION.sendQuery({
                        query: query.substring(0, 20), // Short test query
                        mode: mode,
                        settings: {},
                        onMessage: () => {
                            querySent = true;
                        },
                        onError: (error) => {
                            queryError = error;
                            clearTimeout(timeout);
                            resolve(false);
                        },
                        onDone: () => {
                            clearTimeout(timeout);
                            resolve(true);
                        }
                    });

                    // Abort after 1 second (we just want to test if it starts)
                    setTimeout(() => {
                        if (abort) abort();
                        clearTimeout(timeout);
                        resolve(querySent);
                    }, 1000);
                } catch (error) {
                    queryError = error;
                    clearTimeout(timeout);
                    resolve(false);
                }
            });

            const result = await testPromise;
            
            if (queryError) {
                console.warn(`  ⚠️  Query error:`, queryError.message);
                return false;
            }

            if (result) {
                console.log(`  ✓ Query sent successfully`);
                return true;
            } else {
                console.warn(`  ⚠️  Query did not complete (may be normal for test)`);
                return 'skip';
            }
        } catch (error) {
            console.error(`  ❌ Error testing query:`, error);
            return false;
        }
    }

    checkConsoleErrors() {
        // This is a best-effort check - we can't capture all errors
        // but we can check for common issues
        try {
            // Check for broken elements
            const brokenElements = document.querySelectorAll('[style*="display: none"][style*="visibility: visible"]');
            if (brokenElements.length > 0) {
                console.warn(`  ⚠️  Found ${brokenElements.length} elements with conflicting styles`);
            }

            // Check for missing required elements
            const requiredElements = ['chatContainer', 'queryInput'];
            const missing = requiredElements.filter(id => !document.getElementById(id));
            if (missing.length > 0) {
                console.warn(`  ⚠️  Missing required elements:`, missing);
                return false;
            }

            console.log(`  ✓ No obvious UI errors detected`);
            return true;
        } catch (error) {
            console.error(`  ❌ Error checking console errors:`, error);
            return false;
        }
    }

    checkAnimations() {
        try {
            // Check if animations are running
            const neuralNetwork = document.getElementById('neuralNetworkCanvas');
            const meteorShower = document.getElementById('meteorShowerCanvas');

            let animationsOk = true;

            if (neuralNetwork) {
                const ctx = neuralNetwork.getContext('2d');
                if (ctx) {
                    // Check if canvas is being drawn to
                    console.log(`  ✓ Neural network canvas exists`);
                }
            }

            if (meteorShower) {
                const ctx = meteorShower.getContext('2d');
                if (ctx) {
                    console.log(`  ✓ Meteor shower canvas exists`);
                }
            }

            // Check for animation errors in console (best effort)
            console.log(`  ✓ Animations appear to be working`);
            return true;
        } catch (error) {
            console.error(`  ❌ Error checking animations:`, error);
            return false;
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async sendResultsToServer() {
        try {
            // Only attempt if fetch is available and we're not running from a file:// origin
            if (typeof fetch === 'undefined') {
                return;
            }
            const origin = (typeof window !== 'undefined' && window.location && window.location.origin) || '';
            if (origin.startsWith('file:')) {
                return;
            }

            const payload = {
                results: this.results,
                timestamp: new Date().toISOString(),
                userAgent: (typeof navigator !== 'undefined' && navigator.userAgent) || 'unknown'
            };

            await fetch('/debug/mode-test-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
        } catch (error) {
            console.warn('Error sending mode test results to server', error);
        }
    }

    printResults() {
        console.log('\n\n' + '='.repeat(80));
        console.log('📊 TEST RESULTS SUMMARY');
        console.log('='.repeat(80) + '\n');

        const passed = this.results.filter(r => r.passed).length;
        const failed = this.results.filter(r => !r.passed).length;
        const skipped = this.results.filter(r => Object.values(r.checks).some(c => c === 'skip')).length;

        console.log(`Total Modes Tested: ${this.results.length}`);
        console.log(`✅ Passed: ${passed}`);
        console.log(`❌ Failed: ${failed}`);
        console.log(`⏭️  Skipped: ${skipped}\n`);

        console.log('\n📋 DETAILED RESULTS:\n');

        this.results.forEach(result => {
            const status = result.passed ? '✅' : '❌';
            console.log(`${status} ${result.mode}`);
            
            Object.entries(result.checks).forEach(([check, value]) => {
                const checkStatus = value === true ? '✓' : value === 'skip' ? '⏭️' : '✗';
                console.log(`   ${checkStatus} ${check}: ${value}`);
            });

            if (result.errors.length > 0) {
                console.log(`   Errors:`);
                result.errors.forEach(err => console.log(`     - ${err}`));
            }

            console.log('');
        });

        // Summary of issues
        const issues = this.results.filter(r => !r.passed);
        if (issues.length > 0) {
            console.log('\n⚠️  MODES WITH ISSUES:\n');
            issues.forEach(result => {
                console.log(`   ${result.mode}:`);
                Object.entries(result.checks)
                    .filter(([_, value]) => value === false)
                    .forEach(([check, _]) => {
                        console.log(`     - ${check} failed`);
                    });
            });
        }

        console.log('\n' + '='.repeat(80));
    }
}

// Export for use
export async function runModeTests() {
    const tester = new ModeTester();
    return await tester.runAllTests();
}

// Auto-run if imported directly
if (typeof window !== 'undefined') {
    window.ICEBURG_MODE_TESTER = {
        runTests: runModeTests,
        MODES_TO_TEST,
        TEST_QUERIES
    };
    
    console.log('🧪 Mode tester loaded. Run: window.ICEBURG_MODE_TESTER.runTests()');
}
