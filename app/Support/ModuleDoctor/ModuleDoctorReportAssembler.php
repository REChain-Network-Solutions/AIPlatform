<?php

namespace App\Support\ModuleDoctor;

class ModuleDoctorReportAssembler
{
    public function build(array $analysis): array
    {
        return [
            'summary' => $analysis['message'] ?? null,
            'visibility' => $analysis['visibility_report']['final']['detail'] ?? null,
            'repair_queue_count' => count($analysis['repair_queue'] ?? []),
            'checks' => count($analysis['checks'] ?? []),
            'warnings' => count($analysis['warnings'] ?? []),
            'blocking_issues' => count($analysis['blocking_issues'] ?? []),
        ];
    }
}
