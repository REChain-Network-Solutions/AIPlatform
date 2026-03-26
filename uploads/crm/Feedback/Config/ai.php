<?php

return [
    'provider' => env('AI_PROVIDER', 'openai'),
    'openai' => [
        'api_key' => env('OPENAI_API_KEY'),
        'base' => env('OPENAI_BASE', null),
        'model' => env('OPENAI_MODEL', 'gpt-4o-mini'),
    ],
    'features' => [
        'feedback_generate' => env('AI_FEATURE_FEEDBACK_GENERATE', true),
        'reply_suggest' => env('AI_FEATURE_REPLY_SUGGEST', true),
    ],
];
