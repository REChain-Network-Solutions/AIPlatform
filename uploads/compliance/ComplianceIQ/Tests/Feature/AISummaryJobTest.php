<?php

namespace Modules\ComplianceIQ\Tests\Feature;

use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;
use Modules\ComplianceIQ\Entities\ComplianceReport;
use Modules\ComplianceIQ\Jobs\GenerateComplianceSummary;

class AISummaryJobTest extends TestCase
{
    use RefreshDatabase;

    /** @test */
    public function ai_summary_job_writes_summary_payload()
    {
        $r = ComplianceReport::create([
            'title' => 'Monthly',
            'period_start' => now()->subMonth()->toDateString(),
            'period_end' => now()->toDateString(),
            'status' => 'draft',
        ]);

        (new GenerateComplianceSummary($r->id))->handle();

        $r->refresh();
        $this->assertIsArray($r->summary);
        $this->assertArrayHasKey('overview', $r->summary);
    }
}
