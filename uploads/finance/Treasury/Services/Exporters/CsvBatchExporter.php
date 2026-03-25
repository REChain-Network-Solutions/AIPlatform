<?php
namespace Modules\Treasury\Services\Exporters;
use Modules\Treasury\Models\PaymentRun;

class CsvBatchExporter {
  public function render(PaymentRun $run): string {
    $out = fopen('php://temp', 'r+');
    fputcsv($out, ['beneficiary','amount','reference']);
    foreach ($run->lines as $ln) {
      fputcsv($out, [$ln->beneficiary, number_format($ln->amount,2,'.',''), $ln->reference]);
    }
    rewind($out);
    return stream_get_contents($out);
  }
}
