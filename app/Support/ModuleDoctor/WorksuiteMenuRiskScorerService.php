<?php

namespace App\Support\ModuleDoctor;

class WorksuiteMenuRiskScorerService
{
    public function score(array $visibilityReport): array
    {
        $risk = 0;
        $risk += count($visibilityReport['known_issues'] ?? []) * 10;
        $risk += !empty($visibilityReport['routes']['registered_audit']['missing_route_names'] ?? []) ? 25 : 0;
        $risk += !empty($visibilityReport['permissions']['db_audit']['missing'] ?? []) ? 20 : 0;
        $risk += empty($visibilityReport['package_bridge']['db_entitlements']['packages'] ?? []) ? 20 : 0;
        $risk += empty($visibilityReport['package_bridge']['company_module_settings']['rows'] ?? []) ? 15 : 0;

        return [
            'status' => 'pass',
            'detail' => 'Calculated Worksuite menu visibility risk score.',
            'score' => min($risk, 100),
            'band' => $risk >= 60 ? 'high' : ($risk >= 30 ? 'medium' : 'low'),
        ];
    }
}
