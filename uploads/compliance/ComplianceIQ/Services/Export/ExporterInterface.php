<?php
namespace Modules\ComplianceIQ\Services\Export;
use Modules\ComplianceIQ\Entities\ComplianceReport;
interface ExporterInterface {
    /** @return array [string $filename, string $mime, string $binary] */
    public function export(ComplianceReport $report): array;
}
