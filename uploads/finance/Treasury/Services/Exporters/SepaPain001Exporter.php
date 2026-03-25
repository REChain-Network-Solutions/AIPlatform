<?php
namespace Modules\Treasury\Services\Exporters;
use Modules\Treasury\Models\PaymentRun;

class SepaPain001Exporter {
  public function render(PaymentRun $run): string {
    $dt = date('Y-m-d\TH:i:s');
    $xml = [];
    $xml[] = '<?xml version="1.0" encoding="UTF-8"?>';
    $xml[] = '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.03">';
    $xml[] = '<CstmrCdtTrfInitn><GrpHdr><MsgId>RUN'.$run->id.'</MsgId><CreDtTm>'.$dt.'</CreDtTm><NbOfTxs>'.count($run->lines).'</NbOfTxs></GrpHdr><PmtInf>';
    foreach ($run->lines as $ln) {
      $xml[] = '<CdtTrfTxInf><Amt><InstdAmt Ccy="AUD">'.number_format($ln->amount,2,'.','').'</InstdAmt></Amt><RmtInf><Ustrd>'.htmlspecialchars($ln->reference ?? 'Payment').'</Ustrd></RmtInf></CdtTrfTxInf>';
    }
    $xml[] = '</PmtInf></CstmrCdtTrfInitn></Document>';
    return implode('', $xml);
  }
}
