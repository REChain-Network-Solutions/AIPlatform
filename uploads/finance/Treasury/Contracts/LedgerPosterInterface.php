<?php
namespace Modules\Treasury\Contracts;
interface LedgerPosterInterface {
  public function postBankFees(array $data): ?string;   // returns journal_id
  public function postPayments(array $data): ?string;
  public function postInterest(array $data): ?string;
}
