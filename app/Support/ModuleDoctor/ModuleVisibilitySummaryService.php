<?php

namespace App\Support\ModuleDoctor;

class ModuleVisibilitySummaryService
{
    public function summarize(array $visibility): array
    {
        return [
            'status' => $visibility['final']['status'] ?? 'warn',
            'detail' => $visibility['final']['detail'] ?? 'Visibility not evaluated.',
            'visible_profiles' => collect($visibility['tenant'] ?? [])->filter(fn ($row) => ($row['visible'] ?? false) === true)->keys()->values()->all(),
            'hidden_profiles' => collect($visibility['tenant'] ?? [])->reject(fn ($row) => ($row['visible'] ?? false) === true)->keys()->values()->all(),
        ];
    }
}
