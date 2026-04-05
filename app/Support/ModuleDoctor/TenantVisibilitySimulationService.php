<?php

namespace App\Support\ModuleDoctor;

class TenantVisibilitySimulationService
{
    public function simulate(array $audit): array
    {
        $profiles = app(VisibilityProfileMatrixService::class)->profiles();
        $results = [];

        foreach ($profiles as $profile => $constraints) {
            $results[$profile] = [
                'visible' => $this->visibleFor($audit, $constraints),
                'blockers' => app(TenantVisibilityExplainerService::class)->explain($audit, $constraints),
            ];
        }

        return $results;
    }

    protected function visibleFor(array $audit, array $constraints): bool
    {
        return ($audit['sidebar']['ready'] ?? false)
            && ($audit['routes']['ready'] ?? false)
            && ($audit['permissions']['ready'] ?? false)
            && ($audit['package']['ready'] ?? false)
            && ($audit['menu']['ready'] ?? false)
            && ($audit['provider']['ready'] ?? false)
            && ($constraints['requires_package'] ? ($audit['package']['bridge']['ready'] ?? false) : true);
    }
}
