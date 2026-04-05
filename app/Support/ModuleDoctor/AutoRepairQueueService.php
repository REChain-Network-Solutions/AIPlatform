<?php

namespace App\Support\ModuleDoctor;

class AutoRepairQueueService
{
    public function build(array $audit): array
    {
        $queue = [];

        if (!($audit['permissions']['ready'] ?? false)) {
            $queue[] = ['type' => 'seed_permissions', 'safe' => true];
            $queue[] = ['type' => 'attach_permissions_to_admin', 'safe' => true];
        }

        if (!($audit['package']['bridge']['ready'] ?? false)) {
            $queue[] = ['type' => 'create_package_bridge', 'safe' => true];
        }

        if (!($audit['package']['company']['ready'] ?? false)) {
            $queue[] = ['type' => 'create_company_module_settings', 'safe' => true];
        }

        if (!($audit['menu']['ready'] ?? false) && ($audit['routes']['ready'] ?? false)) {
            $queue[] = ['type' => 'generate_menu_row', 'safe' => true];
        }

        if (($audit['sidebar']['has_sidebar'] ?? false) && ($audit['sidebar']['uses_foreign_permission_api'] ?? false)) {
            $queue[] = ['type' => 'repair_sidebar_permission_patterns', 'safe' => true];
        }

        return $queue;
    }

    /**
     * Alias for build() — called by controller and other services.
     */
    public function queue(array $audit): array
    {
        return $this->build($audit);
    }
}
