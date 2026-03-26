<?php

return [
    /*
     | Titan AI Bridge
     | This module can emit signals and request drafts via TitanZero.
     | Safe by default: bridge is inert if TitanZero isn't installed/enabled.
     */
    'titan' => [
        'enabled' => true,
        'service' => \Modules\TitanZero\Services\ZeroGateway::class,
    ],
];
