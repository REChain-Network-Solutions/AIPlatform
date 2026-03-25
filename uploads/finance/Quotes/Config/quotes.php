<?php

return [
    'number_prefix' => env('QUOTES_PREFIX', 'Q-'),
    // Number pattern tokens: {PREFIX}, {YYYY}, {YY}, {MM}, {DD}, {NN}, {NNN}, {NNNN}, {NNNNN}
    'number_pattern' => env('QUOTES_NUMBER_PATTERN', '{PREFIX}{YYYY}-{NNNNN}'),
    'sequence_series' => env('QUOTES_SERIES', 'default'),
    'default_currency' => env('QUOTES_CURRENCY', 'USD'),
    'tax_inclusive' => env('QUOTES_TAX_INCLUSIVE', false),
    'auto_convert_on_accept' => env('QUOTES_AUTO_CONVERT_ON_ACCEPT', false),
    'notify' => [
        // Comma-separated list of staff emails
        'emails' => env('QUOTES_NOTIFY_EMAILS', ''),
    ],
    'webhook' => [
        'accept_url' => env('QUOTES_WEBHOOK_ACCEPT_URL', ''),
        'secret' => env('QUOTES_WEBHOOK_SECRET', ''),
    ],

    ,
    // Simple branding for public/PDF/email views
    'brand' => [
        'company_name' => env('QUOTES_BRAND_NAME', 'Your Company'),
        'logo_url' => env('QUOTES_BRAND_LOGO', ''),
        'footer_note' => env('QUOTES_BRAND_FOOTER', 'Thank you for your business.'),
        'terms' => env('QUOTES_BRAND_TERMS', ''),
    ],
];

