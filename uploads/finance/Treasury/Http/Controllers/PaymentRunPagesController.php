<?php
namespace Modules\Treasury\Http\Controllers;
use Illuminate\Routing\Controller;
use Modules\Treasury\Models\PaymentRun;

class PaymentRunPagesController extends Controller
{
  public function index() { $runs = PaymentRun::latest()->paginate(20); return view('Treasury::runs.index', compact('runs')); }
  public function show($id) { $run = PaymentRun::with('lines')->findOrFail($id); return view('Treasury::runs.show', compact('run')); }
}
