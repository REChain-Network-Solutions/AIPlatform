<?php

return [
    // Human name for display (Packages label is handled in core lang files)
    'name' => 'Quality Control',

    /*
    |--------------------------------------------------------------------------
    | Feature flags (module-local)
    |--------------------------------------------------------------------------
    */
    'titan_links' => true,

    /*
    |--------------------------------------------------------------------------
    | Cleaning workflow options
    |--------------------------------------------------------------------------
    | These options let the module integrate into cleaning ops without requiring
    | a hard dependency on a specific Jobs/Issues module.
    |
    | - auto_create_on_job_complete: listens for common "job completed" events
    |   and creates a Quality Check schedule automatically.
    | - create_issue_on_needs_reclean: when a check is marked "Needs Re-Clean",
    |   dispatch an event that other modules (Cleaning Issues) can consume.
    */
    'auto_create_on_job_complete' => false,
    'create_issue_on_needs_reclean' => true,

    // File categories used for proof photos.
    'photo_categories' => ['before', 'after', 'general'],
];
