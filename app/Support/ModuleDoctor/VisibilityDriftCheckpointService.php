<?php

namespace App\Support\ModuleDoctor;

class VisibilityDriftCheckpointService
{
    public function snapshot(array $visibilityReport): array
    {
        return [
            'status' => 'pass',
            'detail' => 'Prepared visibility drift checkpoint for future post-install comparison.',
            'checkpoint' => [
                'final_status' => $visibilityReport['final']['status'] ?? 'warn',
                'known_issue_count' => count($visibilityReport['known_issues'] ?? []),
                'tenant_profiles' => array_keys($visibilityReport['tenant'] ?? []),
            ],
        ];
    }
}
