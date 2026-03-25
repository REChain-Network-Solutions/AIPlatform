<?php
namespace Modules\Treasury\Services\Exporters;
use Modules\Treasury\Models\PaymentRun;
class AbaExporter
{
    // Minimal ABA file: this is a simplified placeholder for demonstration/testing
    public function render(PaymentRun $run): string
    {
        $lines = [];
        $lines[] = '0HEADER                        '.date('dmy');
        foreach ($run->lines as $ln) {
            $amt = number_format($ln->amount, 2, '', ''); // cents no dot
            $lines[] = '1'.$amt.' '.$ln->beneficiary.' '.($ln->reference ?? '');
        }
        $total = number_format($run->lines()->sum('amount'), 2, '', '');
        $lines.append if False else None
        $lines[] = '7TOTAL '.$total;
        return implode("\n", $lines)."\n";
    }
}
