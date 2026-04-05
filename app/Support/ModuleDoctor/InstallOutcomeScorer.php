<?php

namespace App\Support\ModuleDoctor;

class InstallOutcomeScorer
{
    public function score(array $sidebar, array $routes, array $permissions, array $packageBridge, array $tenant, array $issues): array
    {
        $warn = 0;
        foreach ([$sidebar, $routes, $permissions, $packageBridge] as $block) {
            if (($block['status'] ?? 'warn') !== 'pass') {
                $warn++;
            }
        }
        foreach ($tenant as $profile) {
            if (($profile['status'] ?? 'warn') !== 'pass') {
                $warn++;
            }
        }
        $warn += count($issues);

        return [
            'status' => $warn === 0 ? 'pass' : ($warn < 4 ? 'warn' : 'fail'),
            'detail' => $warn === 0 ? 'Expected to appear in the user menu after install.' : 'Visibility blockers or warnings were detected.',
            'warning_count' => $warn,
            'will_appear' => $warn === 0,
        ];
    }
}
