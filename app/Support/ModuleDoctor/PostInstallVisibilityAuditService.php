<?php

namespace App\Support\ModuleDoctor;

class PostInstallVisibilityAuditService
{
    public function run(array $module): array
    {
        $audit = app(VisibilityAuditService::class)->audit($module);

        return [
            'audit' => $audit,
            'simulation' => app(TenantVisibilitySimulationService::class)->simulate($audit),
            'repair_queue' => app(AutoRepairQueueService::class)->build($audit),
            'score' => app(WorksuiteMenuRiskScorerService::class)->score($audit),
        ];
    }
}
