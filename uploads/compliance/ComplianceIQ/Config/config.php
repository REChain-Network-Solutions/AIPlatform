<?php

return [
    'name' => 'ComplianceIQ',
    'report_export' => [
        'default' => 'csv',
    ],
    'ai' => [
        'driver' => env('COMPLIANCEIQ_AI_DRIVER', 'null') // 'null' | 'openai'
    ],

    'hash' => [
        'algo' => 'sha256',
    ],
    'schedule' => [
        'report_cron' => '0 2 * * *',
        'tamper_check_cron' => '15 2 * * *',
    ],
];
