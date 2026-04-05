<?php
namespace App\Support\ModuleDoctor;
class MenuEligibilityScannerService {
    public function scan(array $audit): array {
        return [
            'route_exists' => $audit['routes']['ready'] ?? false,
            'provider_booted' => $audit['provider']['ready'] ?? false,
            'menu_registered' => $audit['menu']['ready'] ?? false,
            'permission_chain' => $audit['permissions']['ready'] ?? false,
            'tenant_enabled' => $audit['package']['ready'] ?? false,
        ];
    }
}
