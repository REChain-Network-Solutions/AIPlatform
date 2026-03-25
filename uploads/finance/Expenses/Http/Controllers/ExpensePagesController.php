<?php
namespace Modules\Expenses\Http\Controllers;
use Illuminate\Routing\Controller;
use Modules\Expenses\Models\Expense;
use Modules\Expenses\Models\Receipt;

class ExpensePagesController extends Controller
{
  public function show($id) {
    $e = Expense::findOrFail($id);
    $receipts = Receipt::where('expense_id',$e->id)->get();
    return view('Expenses::show', compact('e','receipts'));
  }
}
