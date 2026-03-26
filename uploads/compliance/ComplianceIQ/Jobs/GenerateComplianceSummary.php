<?php

namespace Modules\ComplianceIQ\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Modules\ComplianceIQ\Entities\ComplianceReport;
use Modules\ComplianceIQ\Services\AI\ComplianceAIInterface;

class GenerateComplianceSummary implements ShouldQueue
{
    protected ComplianceAIInterface $ai;
    use Dispatchable, Queueable;

    public function __construct(public int $reportId) {}

    public function handle(): void
    {
        $this->ai = app(ComplianceAIInterface::class);
        $report = ComplianceReport::find($this->reportId);
        if (!$report) return;

        $summary = $this->ai->summarize($report);

        $report->update(['summary' => $summary]);
    }
}
