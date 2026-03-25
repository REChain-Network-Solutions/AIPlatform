<?php

namespace Modules\EInvoice\Helper;

use App\Models\Invoice;
use Modules\EInvoice\Helper\XmlGenerator\Romania;

class InvoiceXmlGenerate
{

    public static function generateXml(Invoice $invoice)
    {
        return match ($invoice->company->name) {
            'ROMANIA' => Romania::generate($invoice),
            default => Romania::generate($invoice)
        };
    }

}
