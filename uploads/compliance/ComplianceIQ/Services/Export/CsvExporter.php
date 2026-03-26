<?php
namespace Modules\ComplianceIQ\Services\Export;
use Modules\ComplianceIQ\Entities\ComplianceReport;
class CsvExporter implements ExporterInterface {
    public function export(ComplianceReport $report): array {
        $csv = implode(",", ['id','title','period_start','period_end','status'])."\n";
        $csv .= implode(",", [$report->id,$report->title,$report->period_start,$report->period_end,$report->status])."\n";
        $name = "report_{$report->id}.csv";
        return [$name, 'text/csv', $csv];
    }
}
