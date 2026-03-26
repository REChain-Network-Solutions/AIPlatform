<?php

return [
    /*
    |--------------------------------------------------------------------------
    | Automation & AI Integration Manifest (Module-local)
    |--------------------------------------------------------------------------
    |
    | This file is intentionally safe-by-default. It does not execute anything
    | by itself; it merely declares what this module *can* emit/accept.
    |
    | - signals: domain events this module emits (for AI/automation)
    | - actions: proposal/action types this module can handle
    | - autopilot: action classes used by policy engine (A/B/C)
    |
    */

    'signals' => [
        // Example:
        // 'booking_late' => [
        //     'severity' => 'amber',
        //     'facts' => ['minutes_late', 'booking_id', 'customer_id', 'provider_id'],
        // ],
    ],

    'actions' => [
        // Example:
        // 'send_delay_message' => [
        //     'risk' => 'green',
        //     'handler' => null, // set to FQCN when you implement a handler
        //     'schema' => [
        //         'booking_id' => 'int|required',
        //         'channel' => 'string|in:sms,email,push',
        //     ],
        // ],
    ],

    'autopilot' => [
        // Class A: safe auto-execute (tagging, drafting, internal tasks)
        'class_a' => [],
        // Class B: bounded auto-execute (customer comms, assignment within caps)
        'class_b' => [],
        // Class C: always approval (money, cancellation, destructive actions)
        'class_c' => [],
    ],
];
