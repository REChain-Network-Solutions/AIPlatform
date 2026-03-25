<?php

namespace Modules\BudgetAllocationAprovalModule\Http\Controllers;

use Illuminate\Routing\Controller;

class BudgetAllocationAprovalModuleController extends Controller
{
    public function index()
    {
        return view('budgetallocationaprovalmodule::index');
    }
}
