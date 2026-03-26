<?php
/**
 * Module Health Checks — generated for Worksuite compatibility.
 * Returns a list of checks that a Doctor tool (or humans) can run.
 */
return [
    [
        'id' => 'reviewmodule:module_json',
        'label' => 'module.json present',
        'severity' => 'error',
        'ok' => file_exists(__DIR__ . '/../module.json'),
        'hint' => 'module.json is required for module discovery.',
    ],
    [
        'id' => 'reviewmodule:service_provider',
        'label' => 'ServiceProvider present',
        'severity' => 'error',
        'ok' => (bool) glob(__DIR__ . '/../Providers/*ServiceProvider.php'),
        'hint' => 'Ensure Providers/<Module>ServiceProvider.php exists and is registered in module.json.',
    ],
    [
        'id' => 'reviewmodule:routes_web',
        'label' => 'Routes/web.php present',
        'severity' => 'warn',
        'ok' => file_exists(__DIR__ . '/../Routes/web.php') || file_exists(__DIR__ . '/../routes/web.php'),
        'hint' => 'If this module has UI or HTTP endpoints, add Routes/web.php (or routes/web.php for package-style modules).',
    ],
    [
        'id' => 'reviewmodule:migrations_dir',
        'label' => 'Database migrations directory present',
        'severity' => 'warn',
        'ok' => is_dir(__DIR__ . '/../Database/Migrations') || is_dir(__DIR__ . '/../database/migrations'),
        'hint' => 'If this module stores data, ensure it ships migrations under Database/Migrations (or database/migrations).',
    ],
    [
        'id' => 'reviewmodule:views_dir',
        'label' => 'Views directory present',
        'severity' => 'info',
        'ok' => is_dir(__DIR__ . '/../Resources/views') || is_dir(__DIR__ . '/../resources/views'),
        'hint' => 'Optional: add views if module renders Blade UI.',
    ],
    [
        'id' => 'reviewmodule:lang_dir',
        'label' => 'Translations directory present',
        'severity' => 'info',
        'ok' => is_dir(__DIR__ . '/../Resources/lang') || is_dir(__DIR__ . '/../resources/lang'),
        'hint' => 'Optional: add translations if module has user-facing strings.',
    ],

[
    'key' => 'automation_config',
    'label' => 'Automation manifest present',
    'ok' => (is_file(__DIR__ . '/../Config/automation.php') || is_file(__DIR__ . '/../config/automation.php')),
    'hint' => 'Add Config/automation.php (or config/automation.php) to declare signals/actions for AI & automation.',
],
    [
        'id' => 'titan_zero_available',
        'label' => 'TitanZero available',
        'status' => class_exists('\Modules\TitanZero\Services\ZeroGateway'),
        'severity' => 'warning',
        'help' => 'Install/enable TitanZero to use AI bridge features.',
    ],

];
