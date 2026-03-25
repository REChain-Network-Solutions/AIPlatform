<?php
namespace Modules\Expenses\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Http\Request;
use Modules\Expenses\Models\Expense; use Modules\Expenses\Models\ExpenseCategory;
class ExpensesController extends Controller{public function index(){ $rows=Expense::latest()->paginate(25); return view('Expenses::index',compact('rows')); }
public function create(){ $cats=ExpenseCategory::all(); return view('Expenses::create',compact('cats')); }
public function store(Request $r){ $data=$r->validate(['amount'=>'required|numeric','expense_category'=>'nullable|integer','description'=>'nullable|string','date'=>'required|date']); Expense::create($data + ['created_by'=>optional($r->user())->id]); return redirect()->route('expenses.index')->with('ok','Expense recorded'); }}