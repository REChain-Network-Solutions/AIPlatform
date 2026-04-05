<?php

namespace App\Support\ModuleDoctor;

class ModuleDoctorManifest
{
    public function build(): array
    {
        return [
            'workspace' => 'worksuite',
            'pass' => 11,
            'features' => [
                'visibility_report',
                'tenant_simulation',
                'known_issue_rules',
                'auto_repair_queue',
                'protected_paths',
                'registered_route_audit',
                'sidebar_inclusion_audit',
                'permission_db_audit',
                'package_entitlement_audit',
            ],
        ];
    }
}
