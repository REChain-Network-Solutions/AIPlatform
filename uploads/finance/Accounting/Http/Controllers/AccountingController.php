<?php

namespace Modules\Accounting\Http\Controllers;

use App\Charts\CommonChart;
use App\Utils\ModuleUtil;
use DB;
use Illuminate\Http\Response;
use Illuminate\Routing\Controller;
use Modules\Accounting\Entities\AccountingAccount;
use Modules\Accounting\Entities\AccountingAccountType;
use Modules\Accounting\Utils\AccountingUtil;

class AccountingController extends Controller
{
    protected $accountingUtil;

    /**
     * Constructor
     *
     * @return void
     */
    public function __construct(AccountingUtil $accountingUtil, ModuleUtil $moduleUtil)
    {
        $this->accountingUtil = $accountingUtil;
        $this->moduleUtil = $moduleUtil;
    }

    /**
     * Display dashboard for accounting module.
     *
     * @return Response
     */
    public function dashboard()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module'))) {
            abort(403, 'Unauthorized action.');
        }

        $start_date = request()->get('start_date', session()->get('financial_year.start'));
        $end_date = request()->get('end_date', session()->get('financial_year.end'));
        $balance_formula = $this->accountingUtil->balanceFormula();

        $coa_overview = AccountingAccount::leftjoin('accounting_accounts_transactions as AAT',
                                    'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                                ->where('business_id', $business_id)
                                ->whereDate('AAT.operation_date', '>=', $start_date)
                                ->whereDate('AAT.operation_date', '<=', $end_date)
                                ->select(
                                    DB::raw($balance_formula),
                                    'accounting_accounts.account_primary_type'
                                )
                                ->groupBy('accounting_accounts.account_primary_type')
                                ->get();

        $account_types = AccountingAccountType::accounting_primary_type();

        $labels = [];
        $values = [];

        foreach ($account_types as $k => $v) {
            $value = 0;

            foreach ($coa_overview as $overview) {
                if ($overview->account_primary_type == $k && ! empty($overview->balance)) {
                    $value = (float) $overview->balance;
                }
            }
            $values[] = abs($value);

            //Suffix CR/DR as per value
            $tmp = $v['label'];
            if ($value < 0) {
                $tmp .= (in_array($v['label'], ['Asset', 'Expenses']) ? ' (CR)' : ' (DR)');
            }
            $labels[] = $tmp;
        }

        $colors = ['#E75E82', '#37A2EC', '#FACD56', '#5CA85C', '#605CA8',
            '#2f7ed8', '#0d233a', '#8bbc21', '#910000', '#1aadce',
            '#492970', '#f28f43', '#77a1e5', '#c42525', '#a6c96a', ];
        $coa_overview_chart = new CommonChart;
        $coa_overview_chart->labels($labels)
                    ->options($this->__chartOptions())
                    ->dataset(__('accounting::lang.current_balance'), 'pie', $values)
                    ->color($colors);

        $all_charts = [];
        foreach ($account_types as $k => $v) {
            $sub_types = AccountingAccountType::where('account_primary_type', $k)
                                        ->where(function ($q) use ($business_id) {
                                            $q->whereNull('business_id')
                                                ->orWhere('business_id', $business_id);
                                        })
                                        ->get();

            $balances = AccountingAccount::leftjoin('accounting_accounts_transactions as AAT',
                                        'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                                ->where('business_id', $business_id)
                                ->whereDate('AAT.operation_date', '>=', $start_date)
                                ->whereDate('AAT.operation_date', '<=', $end_date)
                                ->select(
                                    DB::raw($balance_formula),
                                    'accounting_accounts.account_sub_type_id'
                                )
                                ->groupBy('accounting_accounts.account_sub_type_id')
                                ->get();

            $labels = [];
            $values = [];

            foreach ($sub_types as $st) {
                $labels[] = $st->account_type_name;
                $value = 0;

                foreach ($balances as $bal) {
                    if ($bal->account_sub_type_id == $st->id && ! empty($bal->balance)) {
                        $value = (float) $bal->balance;
                    }
                }
                $values[] = $value;
            }
            $chart = new CommonChart;
            $chart->labels($labels)
                ->options($this->__chartOptions())
                 ->dataset(__('accounting::lang.current_balance'), 'pie', $values)
                 ->color($colors);

            $all_charts[$k] = $chart;
        }

        return view('accounting::accounting.dashboard')->with(compact('coa_overview_chart',
                'all_charts', 'coa_overview', 'account_types', 'end_date', 'start_date'));
    }

    private function __chartOptions()
    {
        return [
            'plotOptions' => [
                'pie' => [
                    'allowPointSelect' => true,
                    'cursor' => 'pointer',
                    'dataLabels' => [
                        'enabled' => false,
                    ],
                    'showInLegend' => true,
                ],
            ],
        ];
    }

    public function AccountsDropdown()
    {
        if (request()->ajax()) {
            $business_id = request()->session()->get('user.business_id');
            $q = request()->input('q', '');
            $accounts = AccountingAccount::forDropdown($business_id, true, $q);

            $accounts_array = [];
            foreach ($accounts as $account) {
                $accounts_array[] = [
                    'id' => $account->id,
                    'text' => $account->name.' - <small class="text-muted">'.$account->account_primary_type.' - '.
                                $account->sub_type.'</small>',
                    'html' => $account->name.' - <small class="text-muted">'.$account->account_primary_type.' - '.
                                $account->sub_type.'</small>',
                ];
            }

            return $accounts_array;
        }
    }
}
