/**
 * ICEBURG Unified Configuration
 * 
 * Consolidates configuration from inline scripts and external `iceburg-config.json`
 * Provides a single `window.ICEBURG_CONFIG` object for the application.
 */

(function () {
    // Default configuration (fallback)
    const DEFAULT_CONFIG = {
        "status": "ready",
        "serverConfig": {
            "iceburg_version": "2.0",
            "iceburg_branding": "Iceberg",
            "iceburg_description": "Truth-Finding AI Civilization",
            "models": [
                {
                    "modelId": "qwen2.5:7b",
                    "name": "ICEBURG Fast",
                    "description": "Quick responses",
                    "modeDescription": "Fast mode for quick answers",
                    "modelMode": "ICEBURG_MODE_FAST",
                    "agent": "auto",
                    "tags": [],
                    "badgeText": "",
                    "isDefault": true
                },
                {
                    "modelId": "deepseek-v2:16b",
                    "name": "ICEBURG Surveyor",
                    "description": "Research Expert",
                    "modeDescription": "Deep research and gnostic analysis",
                    "modelMode": "ICEBURG_MODE_SURVEYOR",
                    "agent": "surveyor",
                    "tags": ["research", "gnosis"],
                    "badgeText": ""
                },
                {
                    "modelId": "deepseek-v2:16b",
                    "name": "ICEBURG Protocol",
                    "description": "Multi-Agent",
                    "modeDescription": "Full protocol with all agents",
                    "modelMode": "ICEBURG_MODE_PROTOCOL",
                    "agent": "protocol",
                    "tags": ["multi-agent"],
                    "badgeText": ""
                }
            ],
            "unavailableModels": [],
            "defaultFastModelId": "qwen2.5:7b",
            "defaultExpertModelId": "deepseek-v2:16b",
            "defaultProtocolModelId": "deepseek-v2:16b",
            "agents": {
                "agentPresets": [
                    {
                        "agentId": "surveyor",
                        "title": "Surveyor",
                        "systemPrompt": "You are ICEBURG Surveyor...",
                        "icon": "🔬",
                        "description": "Gnostic research agent",
                        "capabilities": ["research", "gnosis"]
                    },
                    // ... other agents will be loaded from server config or added dynamically
                ]
            }
        },
        "thinking_config": {
            "thinking_auto_open": false,
            "enable_single_thinking_different_summary_ui": true,
            "iceburg_thinking_use_js": false,
            "show_show_thoughts": true,
            "thinking_stream_enabled": true,
            "thinking_glitch_animation": true,
            "thinking_dropdown_enabled": true
        },
        "streaming_config": {
            "streaming_markdown_config": {
                "cutLength": 50,
                "maxHoldTimeMs": 300,
                "isEnabled": true,
                "wordBoundaryChunking": true
            },
            "chunk_delay": 0.02,
            "thinking_stream_poll_interval": 50
        },
         "timeline_navigator": {
            "enabled": true,
            "maxResponses": 100,
            "minResponses": 2,
            "minScreenWidth": 768,
            "highlightOnScroll": true,
            "showAgentBadges": true
        },
         "response_feedback": {
            "show_like_dropdown": true,
            "show_dislike_dropdown": true,
            "show_research_quality": true,
            "show_accuracy_feedback": true,
            "satisfaction_score": 3,
            "enable_feedback_storage": true
        },
        "feature_flags": {
            "enable_memory_toggle": true,
            "enable_text_to_speech": false,
            "enable_code_execution": true,
            "enable_file_sharing": true,
            "enable_mermaid_diagrams": true,
            "enable_sketchpad": false,
            "enable_iceburg_tasks": false,
            "enable_tool_composer": false,
            "enable_voice_mode": false,
            "enable_image_generation": false
        },
         "suggestions_config": {
            "enabled": true,
            "maxItems": 7,
            "maxItemsMobile": 3,
            "minChars": 1,
            "maxChars": 75,
            "throttleTimeMs": 250
        },
        "typeahead_config": {
            "enabled": true,
            "minChars": 1,
            "maxChars": 40,
            "maxResults": 7,
            "maxResultsMobile": 4,
            "maxWords": 50,
            "throttleTimeMs": 80
        }
    };

    // Initialize global config with defaults
    window.ICEBURG_CONFIG = DEFAULT_CONFIG;

    // Fetch updated config from server/static file
    fetch('/iceburg-config.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Loaded iceburg-config.json');
            // Merge fetched config into global config (shallow merge for now)
            window.ICEBURG_CONFIG = { ...window.ICEBURG_CONFIG, ...data };
            
            // Dispatch event for components listening for config updates
            window.dispatchEvent(new CustomEvent('iceburg-config-updated', { 
                detail: window.ICEBURG_CONFIG 
            }));
        })
        .catch(error => {
            console.warn('⚠️ Could not load iceburg-config.json, using defaults:', error);
             // Dispatch event even if failed (using defaults)
             window.dispatchEvent(new CustomEvent('iceburg-config-updated', { 
                detail: window.ICEBURG_CONFIG 
            }));
        });

    console.log('⚙️ ICEBURG Configuration initialized');
})();
