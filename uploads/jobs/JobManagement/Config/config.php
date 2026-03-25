<?php

return [
    'name' => 'JobManagement',
    // Feature toggles & defaults
    'features' => [
        'work_requests' => true,
        'work_orders' => true,
        'recurring' => true,
        'services_catalog' => true,
        'meters' => true,
        'files' => true,
    ],
    // Default route prefix; change to match your app
    'route_prefix' => '',
    // Menu hint for integrators (Worksuite, QuickAI bridges can read this)
    'menu' => [
        'group' => 'Operations',
        'label' => 'Job Management',
        'icon'  => 'ri-briefcase-4-line',
        'route' => 'engineerings.index',
        'children' => [
            ['label' => 'Work Orders', 'route' => 'work.index'],
            ['label' => 'Recurring', 'route' => 'recurring-work.index'],
            ['label' => 'Services', 'route' => 'engineerings.index'],
            ['label' => 'Meters', 'route' => 'meter.export'],
        ]
    ]
];
