<?php

namespace App\Support\ModuleDoctor;

class CompatibilityRuleEngine
{
    public function evaluate(array $audit): array
    {
        return [
            'worksuite_base' => true,
            'menu_truth_ready' => ($audit['routes']['ready'] ?? false) && ($audit['provider']['ready'] ?? false),
            'tenant_visibility_ready' => ($audit['package']['ready'] ?? false),
            'safe_autofix_available' => count(app(AutoRepairQueueService::class)->build($audit)) > 0,
        ];
    }
}
