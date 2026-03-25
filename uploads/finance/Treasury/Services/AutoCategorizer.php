<?php
namespace Modules\Treasury\Services;
use Modules\Treasury\Models\ReconciliationRule;

class AutoCategorizer {
  public function apply(array &$row): void {
    $rules = ReconciliationRule::all();
    foreach ($rules as $r) {
      $pattern = str_replace('%','.*', preg_quote($r->pattern,'/'));
      if (preg_match('/'.$pattern.'/i', (string)($row['description'] ?? ''))) {
        $row['account_code'] = $r->account_code;
        $row['category'] = $r->direction === 'in' ? 'income' : 'expense';
        break;
      }
    }
  }
}
