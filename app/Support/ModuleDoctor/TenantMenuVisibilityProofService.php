<?php

namespace App\Support\ModuleDoctor;

class TenantMenuVisibilityProofService
{
    public function build(string $profile, array $decision, array $signals): array
    {
        return [
            'profile' => $profile,
            'visible' => (bool) ($decision['visible'] ?? false),
            'proof' => [
                'sidebar_ready' => !empty($signals['sidebar']['menu_ready']),
                'registered_routes_ok' => empty($signals['routes']['registered_audit']['missing_route_names'] ?? []),
                'permissions_seeded' => empty($signals['permissions']['db_audit']['missing'] ?? []),
                'role_mapping_present' => !empty($signals['permissions']['role_matrix']['matrix'] ?? []),
                'package_entitlement_present' => !empty($signals['package_bridge']['db_entitlements']['packages'] ?? []),
                'company_module_settings_present' => !empty($signals['package_bridge']['company_module_settings']['rows'] ?? []),
            ],
            'blockers' => $decision['blockers'] ?? [],
        ];
    }
}
