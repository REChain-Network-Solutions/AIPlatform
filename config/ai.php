<?php

return [
    'routing' => [
        // Preferred execution order for completions
        'order' => [
            'device', // on-device (stub hook)
            'local',  // local/Ollama-compatible (stub hook)
            'cloud',  // existing remote providers
        ],
        // Toggle stubs
        'device_enabled' => env('AI_DEVICE_ENABLED', false),
        'local_enabled'  => env('AI_LOCAL_ENABLED', false),
    ],
];
