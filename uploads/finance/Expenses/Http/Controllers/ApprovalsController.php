<?php
namespace Modules\Expenses\Http\Controllers;
use Illuminate\Routing\Controller;
use Illuminate\Http\Request;
use Modules\Expenses\Models\Expense;

class ApprovalsController extends Controller
{
  public function submit($id) {
    $e = Expense::findOrFail($id);
    $e->status = 'submitted';
    $e->save();
    return redirect()->back()->with('ok','Expense submitted for approval');
  }
  public function approve(Request $r, $id) {
    $e = Expense::findOrFail($id);
    $e->status = 'approved';
    $e->approved_by = optional($r->user())->id;
    $e->approved_at = now();
    $e->save();
    return redirect()->back()->with('ok','Expense approved');
  }
  public function reimburse($id) {
    $e = Expense::findOrFail($id);
    $e->status = 'reimbursed';
    $e->reimbursed_at = now();
    $e->save();
    return redirect()->back()->with('ok','Expense marked reimbursed');
  }
}
