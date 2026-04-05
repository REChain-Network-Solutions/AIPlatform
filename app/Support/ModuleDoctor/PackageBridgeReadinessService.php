<?php

namespace App\Support\ModuleDoctor;

class PackageBridgeReadinessService
{
    public function inspect(array $module): array
    {
        $moduleName = $module['module_name'] ?? (is_string($module) ? $module : '');
        $bridge = app(PackageEntitlementAuditService::class)->inspect($moduleName);
        $company = app(CompanyModuleSettingAuditService::class)->inspect($moduleName);
        $registry = app(ModuleRegistryAuditService::class)->inspect($moduleName);

        return [
            'bridge' => $bridge,
            'company' => $company,
            'registry' => $registry,
            'ready' => ($bridge['status'] ?? 'warn') === 'pass'
                && ($company['status'] ?? 'warn') === 'pass'
                && ($registry['status'] ?? 'warn') === 'pass',
        ];
    }
}
