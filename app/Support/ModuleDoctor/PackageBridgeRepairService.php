<?php

namespace App\Support\ModuleDoctor;

class PackageBridgeRepairService
{
    public function plan(array $bridgeAudit): array
    {
        $needsPackages = empty($bridgeAudit['db_entitlements']['packages'] ?? []);
        $needsSettings = empty($bridgeAudit['company_module_settings']['rows'] ?? []);

        return [
            'status' => ($needsPackages || $needsSettings) ? 'warn' : 'pass',
            'detail' => ($needsPackages || $needsSettings)
                ? 'Package entitlement and tenant module settings can be auto-seeded during install.'
                : 'Package bridge and tenant module settings look ready.',
        ];
    }
}
