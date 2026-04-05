<?php

namespace App\Support\ModuleDoctor;

class DiagnosticsViewModel
{
    public function make(array $module): array
    {
        $audit = app(VisibilityAuditService::class)->audit($module);

        return [
            'audit' => $audit,
            'simulation' => app(TenantVisibilitySimulationService::class)->simulate($audit),
            'repair_queue' => app(AutoRepairQueueService::class)->build($audit),
            'decision' => $audit['decision'] ?? [],
            'compatibility' => app(CompatibilityRuleEngine::class)->evaluate($audit),
        ];
    }
}
