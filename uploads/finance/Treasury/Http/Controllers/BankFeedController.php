<?php
namespace Modules\Treasury\Http\Controllers;
use Illuminate\Routing\Controller;
use Illuminate\Http\Request;
use Modules\Treasury\Services\CsvBankFeedImporter;
use Modules\Treasury\Models\BankTransaction;
use Modules\Treasury\Services\AutoCategorizer;

class BankFeedController extends Controller
{
  public function upload(Request $r) {
    $r->validate(['file'=>'required|file']);
    $raw = file_get_contents($r->file('file')->getRealPath());
    $rows = (new CsvBankFeedImporter())->parse($raw);
    foreach ($rows as $row) {
      (new AutoCategorizer())->apply($row);
      BankTransaction::create([
        'bank_account_id' => $r->get('bank_account_id'),
        'date' => $row['date'],
        'description' => $row['description'],
        'amount' => $row['amount'],
        'direction' => ($row['amount'] ?? 0) >= 0 ? 'in' : 'out',
        'reference' => $row['reference'] ?? null,
        'status' => 'unreconciled',
        'account_code' => $row['account_code'] ?? null,
        'category' => $row['category'] ?? null,
      ]);
    }
    return redirect()->back()->with('ok', 'Bank feed imported: '.count($rows).' transactions');
  }
}
