<?php
namespace Modules\Treasury\Http\Controllers;
use Illuminate\Routing\Controller;
use Illuminate\Http\Request;
use Modules\Treasury\Models\ReconciliationRule;

class RulesController extends Controller
{
    public function index() {
        $rules = ReconciliationRule::orderBy('id','desc')->get();
        return view('Treasury::rules.index', compact('rules'));
    }
    public function store(Request $r) {
        $data = $r->validate([
            'pattern' => ['required','string','max:191'],
            'account_code' => ['nullable','string','max:64'],
            'direction' => ['required','in:in,out']
        ]);
        ReconciliationRule::create($data);
        return redirect()->back()->with('ok','Rule added');
    }
}
