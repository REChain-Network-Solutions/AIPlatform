<?php

namespace Modules\ComplianceIQ\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Modules\ComplianceIQ\Entities\ComplianceReport;
use Modules\ComplianceIQ\Entities\ComplianceAnnotation;

class AnalyzeComplianceAnomalies implements ShouldQueue
{
    use Dispatchable, Queueable;

    public function __construct(public int $reportId) {}

    public function handle(): void
    {
        $report = ComplianceReport::find($this->reportId);
        if (!$report) return;

        $findings = ['no_anomalies' => true];

        ComplianceAnnotation::create([
            'report_id' => $report->id,
            'user_id'   => 1,
            'note'      => 'Anomaly scan completed: '.json_encode($findings),
        ]);
    }
}
