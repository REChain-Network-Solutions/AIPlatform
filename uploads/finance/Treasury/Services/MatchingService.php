<?php
namespace Modules\Treasury\Services;
use Modules\Treasury\Models\BankTransaction;

class MatchingService {
  /** Very naive match on same absolute amount within +/- 2 days and description similarity */
  public function suggest(string $bookJson, string $bankJson): array {
    $book = json_decode($bookJson, true) ?: [];
    $bank = json_decode($bankJson, true) ?: [];
    $score = 0;
    if (isset($book['amount'], $bank['amount'])) {
      $score += (abs(abs($book['amount']) - abs($bank['amount'])) < 0.01) ? 60 : 0;
    }
    if (isset($book['date'], $bank['date'])) {
      $bd = strtotime($book['date']); $kd = strtotime($bank['date']);
      $score += (abs($bd - $kd) <= 2*86400) ? 25 : 0;
    }
    if (!empty($book['description']) && !empty($bank['description'])) {
      similar_text(strtolower($book['description']), strtolower($bank['description']), $pct);
      $score += (int) round($pct * 0.15); // up to 15
    }
    return ['score' => min(100, $score), 'match' => $score >= 75];
  }
}
