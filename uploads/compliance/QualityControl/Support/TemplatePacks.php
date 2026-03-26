<?php

namespace Modules\QualityControl\Support;

class TemplatePacks
{
    /**
     * UI-only scaffolding for future tradie packs (builder/plumber/electrician).
     * Keep compliance in Titan Compliance; these are plain inspection checklists.
     */
    public static function defaultPacks(): array
    {
        return [
            'builder' => ['Pre-start site walk', 'Access & egress', 'Housekeeping', 'PPE compliance'],
            'electrician' => ['RCD testing', 'Isolation & lockout', 'Cable protection'],
            'plumber' => ['Confined space check', 'Hot works', 'Pressure test evidence'],
            'cleaner' => ['Chemical handling', 'Slip hazards', 'PPE and signage'],
        ];
    }
}
