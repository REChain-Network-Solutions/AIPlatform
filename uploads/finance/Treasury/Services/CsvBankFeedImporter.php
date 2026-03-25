<?php
namespace Modules\Treasury\Services;
use Modules\Treasury\Contracts\BankFeedImporterInterface;
class CsvBankFeedImporter implements BankFeedImporterInterface {
  public function parse(string $raw): array {
    $rows = [];
    $lines = preg_split('/\r?\n/', trim($raw));
    foreach ($lines as $i => $line) {
      if ($i === 0 && stripos($line, 'date') !== false && stripos($line, 'amount') !== false) continue; // header
      if (trim($line)==='') continue;
      $cols = str_getcsv($line);
      $rows[] = [
        'date' => $cols[0] ?? date('Y-m-d'),
        'description' => $cols[1] ?? '',
        'amount' => isset($cols[2]) ? (float)$cols[2] : 0.0,
        'reference' => $cols[3] ?? null,
      ];
    }
    return $rows;
  }
}
