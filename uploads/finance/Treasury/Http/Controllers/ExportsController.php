<?php
namespace Modules\Treasury\Http\Controllers;
use Illuminate\Routing\Controller;
use Modules\Treasury\Models\PaymentRun;
use Modules\Treasury\Services\Exporters\AbaExporter;
use Modules\Treasury\Services\Exporters\SepaPain001Exporter;
use Modules\Treasury\Services\Exporters\CsvBatchExporter;
class ExportsController extends Controller
{
    public function aba($id)
    {
        $run = PaymentRun::with('lines')->findOrFail($id);
        $payload = (new AbaExporter())->render($run);
        return response($payload, 200, [
            'Content-Type' => 'text/plain',
            'Content-Disposition' => 'attachment; filename="payment_run_'.$id.'.aba"',
        ]);
    }
}

    public function sepa($id)
    {
        $run = PaymentRun::with('lines')->findOrFail($id);
        $payload = (new SepaPain001Exporter())->render($run);
        return response($payload, 200, [
            'Content-Type' => 'application/xml',
            'Content-Disposition' => 'attachment; filename="payment_run_'.$id.'.xml"',
        ]);
    }
    public function csv($id)
    {
        $run = PaymentRun::with('lines')->findOrFail($id);
        $payload = (new CsvBatchExporter())->render($run);
        return response($payload, 200, [
            'Content-Type' => 'text/csv',
            'Content-Disposition' => 'attachment; filename="payment_run_'.$id.'.csv"',
        ]);
    }
