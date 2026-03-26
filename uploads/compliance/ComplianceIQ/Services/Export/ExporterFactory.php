<?php
namespace Modules\ComplianceIQ\Services\Export;
class ExporterFactory {
    public static function make(?string $driver): ExporterInterface {
        $driver = $driver ?: config('complianceiq.report_export.default', 'csv');
        return match($driver) {
            'pdf' => new PdfExporter(),
            default => new CsvExporter(),
        };
    }
}
