<?php

namespace App\Support\ModuleDoctor\Exports;

class ModuleVisibilityExportService
{
    public function build(array $visibilityReport): array
    {
        return [
            'generated_at' => now()->toIso8601String(),
            'final' => $visibilityReport['final'] ?? [],
            'tenant_matrix' => $visibilityReport['tenant_matrix'] ?? [],
            'known_issues' => $visibilityReport['known_issues'] ?? [],
            'checklist' => $visibilityReport['checklist'] ?? [],
        ];
    }
}
