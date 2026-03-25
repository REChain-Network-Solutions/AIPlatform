<?php

return [
    'require_consent' => env('VOICE_REQUIRE_CONSENT', true),
    'allow_remote_stream' => env('VOICE_ALLOW_REMOTE_STREAM', true),
    'consent_hint' => env('VOICE_CONSENT_HINT', 'Voice streaming sends audio to remote providers. Please confirm to continue.'),
];
