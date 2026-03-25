<?php
namespace Modules\Treasury\Services;
use Modules\Treasury\Models\PaymentRun;
use Modules\Treasury\Models\PaymentLine;

class PaymentRunService {
  public function create(array $payload): PaymentRun {
    $run = PaymentRun::create([
      'scheduled_on' => $payload['scheduled_on'] ?? date('Y-m-d'),
      'status' => 'draft',
      'bank_account_id' => $payload['bank_account_id'] ?? null,
    ]);
    foreach (($payload['lines'] ?? []) as $ln) {
      PaymentLine::create([
        'payment_run_id' => $run->id,
        'beneficiary' => $ln['beneficiary'] ?? 'Unknown',
        'amount' => (float) ($ln['amount'] ?? 0),
        'reference' => $ln['reference'] ?? null,
        'status' => 'pending',
      ]);
    }
    return $run;
  }
}
