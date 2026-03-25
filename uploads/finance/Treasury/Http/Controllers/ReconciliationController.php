<?php
namespace Modules\Treasury\Http\Controllers;
use Illuminate\Routing\Controller;
use Illuminate\Http\Request;
use Modules\Treasury\Services\MatchingService;

class ReconciliationController extends Controller
{
  public function suggest(Request $r) {
    $data = $r->validate(['book_txn'=>'required|string','bank_txn'=>'required|string']);
    $svc = new MatchingService();
    return response()->json(['ok'=>true, 'result'=>$svc->suggest($data['book_txn'], $data['bank_txn'])]);
  }
}
