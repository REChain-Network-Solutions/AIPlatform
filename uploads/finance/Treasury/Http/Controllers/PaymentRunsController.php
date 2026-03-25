<?php
namespace Modules\Treasury\Http\Controllers;
use Illuminate\Routing\Controller;
use Illuminate\Http\Request;
use Modules\Treasury\Services\PaymentRunService;
use Modules\Treasury\Models\PaymentRun;

class PaymentRunsController extends Controller
{
  public function create(Request $r) {
    $payload = $r->validate([
      'scheduled_on' => ['required','date'],
      'bank_account_id' => ['nullable','integer'],
      'lines' => ['array','min:1'],
      'lines.*.beneficiary' => ['required','string'],
      'lines.*.amount' => ['required','numeric'],
      'lines.*.reference' => ['nullable','string'],
    ]);
    $run = (new PaymentRunService())->create($payload);
    return response()->json(['ok'=>true,'run_id'=>$run->id]);
  }

  public function show($id) {
    $run = PaymentRun::with('lines')->findOrFail($id);
    return response()->json(['ok'=>true,'data'=>$run]);
  }
}
