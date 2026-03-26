<?php

namespace Modules\ComplianceIQ\Tests\Feature;

use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;
use Modules\ComplianceIQ\Entities\ComplianceReport;
use App\Models\User;

class ReportGenerationTest extends TestCase
{
    use RefreshDatabase;

    /** @test */
    public function admin_can_create_and_export_report()
    {
        $admin = User::factory()->create();
        $this->actingAs($admin);

        $resp = $this->post('/admin/compliance/reports', [
            'title' => 'Quarterly Q3',
            'period_start' => now()->subMonths(3)->toDateString(),
            'period_end' => now()->toDateString(),
            'filters' => ['template' => 'baseline'],
        ]);
        $resp->assertRedirect();

        $report = ComplianceReport::first();
        $this->assertNotNull($report);

        $exp = $this->post("/admin/compliance/reports/{$report->id}/export");
        $exp->assertOk();
        $exp->assertHeader('content-disposition');
    }
}
