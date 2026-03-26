<?php
namespace Modules\ComplianceIQ\Services\Export;
use Modules\ComplianceIQ\Entities\ComplianceReport;

class PdfExporter implements ExporterInterface {
    public function export(ComplianceReport $report): array {
        $html = view('complianceiq::admin.reports.pdf', compact('report'))->render();
        $binary = null;
        $mime = 'application/pdf';
        $name = "report_{$report->id}.pdf";
        try {
            if (class_exists(\Barryvdh\DomPDF\Facade\Pdf::class)) {
                $pdf = \Barryvdh\DomPDF\Facade\Pdf::loadHTML($html);
                $binary = $pdf->output();
            } elseif (class_exists(\Dompdf\Dompdf::class)) {
                $dompdf = new \Dompdf\Dompdf();
                $dompdf->loadHtml($html);
                $dompdf->render();
                $binary = $dompdf->output();
            }
        } catch (\Throwable $e) {
            $binary = null;
        }
        if (!$binary) {
            // Fallback to HTML download if PDF libs not present
            $mime = 'text/html';
            $name = "report_{$report->id}.html";
            $binary = $html;
        }
        return [$name, $mime, $binary];
    }
}
