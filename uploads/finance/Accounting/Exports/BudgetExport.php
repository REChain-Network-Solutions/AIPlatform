<?php

namespace Modules\Accounting\Exports;

use Illuminate\Contracts\View\View;
use Maatwebsite\Excel\Concerns\FromView;

class BudgetExport implements FromView
{
    protected $accounts;

    protected $budget;

    protected $months;

    protected $fy_year;

    protected $account_types;

    protected $view_type;

    public function __construct($accounts, $budget, $months, $fy_year, $account_types, $view_type)
    {
        $this->accounts = $accounts;
        $this->budget = $budget;
        $this->months = $months;
        $this->fy_year = $fy_year;
        $this->account_types = $account_types;
        $this->view_type = $view_type;
    }

    public function view(): View
    {
        if ($this->view_type == 'monthly') {
            return view('accounting::budget.partials.budget_monthly_excel')->with([
                'accounts' => $this->accounts,
                'budget' => $this->budget,
                'months' => $this->months,
                'account_types' => $this->account_types,
            ]);
        } elseif ($this->view_type == 'quarterly') {
            return view('accounting::budget.partials.budget_quarterly_excel')->with([
                'accounts' => $this->accounts,
                'budget' => $this->budget,
                'months' => $this->months,
                'account_types' => $this->account_types,
            ]);
        } elseif ($this->view_type == 'yearly') {
            return view('accounting::budget.partials.budget_yearly_excel')->with([
                'accounts' => $this->accounts,
                'budget' => $this->budget,
                'months' => $this->months,
                'fy_year' => $this->fy_year,
                'account_types' => $this->account_types,
            ]);
        }
    }
}
