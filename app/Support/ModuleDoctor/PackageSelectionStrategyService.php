<?php

namespace App\Support\ModuleDoctor;

class PackageSelectionStrategyService
{
    public function build(array $packageRows = []): array
    {
        return [
            'status' => count($packageRows) ? 'pass' : 'warn',
            'detail' => count($packageRows)
                ? 'Recommended strategy: attach to selected packages already granting adjacent modules first.'
                : 'No package data was available to recommend a selection strategy.',
            'recommendation' => count($packageRows) ? 'selected-first' : 'review-required',
        ];
    }
}
