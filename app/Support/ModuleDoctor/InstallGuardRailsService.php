<?php

namespace App\Support\ModuleDoctor;

class InstallGuardRailsService
{
    public function build(array $visibilityReport): array
    {
        $blocks = [];
        if (!empty($visibilityReport['routes']['registered_audit']['missing_route_names'] ?? [])) {
            $blocks[] = 'Prevent install-finalize until unresolved sidebar route names are acknowledged.';
        }
        if (!empty($visibilityReport['permissions']['db_audit']['missing'] ?? [])) {
            $blocks[] = 'Require permission seeding or explicit skip acknowledgement.';
        }

        return [
            'status' => count($blocks) ? 'warn' : 'pass',
            'detail' => count($blocks) ? 'Installer guard rails added for visibility blockers.' : 'No extra installer guard rails required.',
            'items' => $blocks,
        ];
    }
}
