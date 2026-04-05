<?php

namespace App\Support\ModuleDoctor;

class TenantVisibilityExplainerService
{
    public function explain(array $signals, string $profile, array $blockers = []): array
    {
        $map = @include __DIR__ . '/Truth/visibility-truth-rules.php';
        $reasons = [];
        foreach ($blockers as $blocker) {
            $reasons[] = $map[$blocker] ?? ('Visibility blocker detected: ' . $blocker);
        }

        if ($profile !== 'super_admin' && empty($signals['menu_registration']['hits'] ?? [])) {
            $reasons[] = 'No module references were found in the live Worksuite menu/sidebar entry files.';
        }

        return [
            'reasons' => $reasons,
            'detail' => count($reasons) ? implode(' ', $reasons) : 'No obvious tenant visibility blocker was detected.',
        ];
    }
}
