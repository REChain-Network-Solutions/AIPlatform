<?php

return [
    /*
    |--------------------------------------------------------------------------
    | Titan Integrations (UI link-outs only)
    |--------------------------------------------------------------------------
    | This module intentionally does not implement SWMS or compliance logic.
    | It provides safe link-outs to Titan Docs (SWMS) and Titan Compliance.
    | All links are guarded with Route::has() and can be overridden via env/config.
    */

    'titan_docs' => [
        // If a named route exists, we use it. Otherwise we fall back to url.
        'route' => env('INSPECTION_TITAN_DOCS_ROUTE', 'titan-docs.index'),
        'url'   => env('INSPECTION_TITAN_DOCS_URL',  '/account/titan-docs'),
        'label' => 'Open in Titan Docs',
    ],

    'titan_compliance' => [
        'route' => env('INSPECTION_TITAN_COMPLIANCE_ROUTE', 'titan-compliance.index'),
        'url'   => env('INSPECTION_TITAN_COMPLIANCE_URL',  '/account/titan-compliance'),
        'label' => 'Open in Titan Compliance',
    ],
];
