<?php

namespace App\Support\ModuleDoctor;

class AuditNarrativeService
{
    public function build(array $visibilityReport): array
    {
        $final = $visibilityReport['final'] ?? [];
        $known = $visibilityReport['known_issues'] ?? [];

        $summary = $final['detail'] ?? 'Visibility audit completed.';
        if (count($known)) {
            $summary .= ' Known blockers detected: ' . collect($known)->pluck('title')->implode(', ') . '.';
        }

        return [
            'status' => 'pass',
            'detail' => 'Built human-readable narrative for visibility audit.',
            'summary' => $summary,
        ];
    }
}
