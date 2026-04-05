<?php
return [
    'storage_path' => storage_path('app/custom-modules'),
    'allowed_extensions' => ['zip'],
    'doctor_cards_enabled' => true,
    'reload_after_upload' => true,
    'results_panel_enabled' => true,
    'registry' => [
        'enabled' => true,
        'http_timeout' => 8,
        'auto_ingest_on_analyze' => false,
        'default_source_type' => 'file',
    ],
];
