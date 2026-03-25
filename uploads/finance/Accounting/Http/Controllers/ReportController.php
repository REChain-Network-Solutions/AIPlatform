<?php

namespace Modules\Accounting\Http\Controllers;

use App\BusinessLocation;
use App\Utils\BusinessUtil;
use App\Utils\ModuleUtil;
use DB;
use Illuminate\Http\Response;
use Illuminate\Routing\Controller;
use Modules\Accounting\Entities\AccountingAccount;
use Modules\Accounting\Utils\AccountingUtil;

class ReportController extends Controller
{
    protected $accountingUtil;

    protected $businessUtil;

    /**
     * Constructor
     *
     * @return void
     */
    public function __construct(AccountingUtil $accountingUtil, BusinessUtil $businessUtil,
    ModuleUtil $moduleUtil)
    {
        $this->accountingUtil = $accountingUtil;
        $this->businessUtil = $businessUtil;
        $this->moduleUtil = $moduleUtil;
    }

    /**
     * Display a listing of the resource.
     *
     * @return Response
     */
    public function index()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        $first_account = AccountingAccount::where('business_id', $business_id)
                            ->where('status', 'active')
                            ->first();
        $ledger_url = null;
        if (! empty($first_account)) {
            $ledger_url = route('accounting.ledger', $first_account);
        }

        return view('accounting::report.index')
            ->with(compact('ledger_url'));
    }

    /**
     * Trial Balance
     *
     * @return Response
     */
    public function trialBalance()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        if (! empty(request()->start_date) && ! empty(request()->end_date)) {
            $start_date = request()->start_date;
            $end_date = request()->end_date;
        } else {
            $fy = $this->businessUtil->getCurrentFinancialYear($business_id);
            $start_date = $fy['start'];
            $end_date = $fy['end'];
        }

        $accounts = AccountingAccount::join('accounting_accounts_transactions as AAT',
                                'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                            ->where('business_id', $business_id)
                            ->whereDate('AAT.operation_date', '>=', $start_date)
                            ->whereDate('AAT.operation_date', '<=', $end_date)
                            ->select(
                                DB::raw("SUM(IF(AAT.type = 'credit', AAT.amount, 0)) as credit_balance"),
                                DB::raw("SUM(IF(AAT.type = 'debit', AAT.amount, 0)) as debit_balance"),
                                'accounting_accounts.name'
                            )
                            ->groupBy('accounting_accounts.name')
                            ->get();

        return view('accounting::report.trial_balance')
            ->with(compact('accounts', 'start_date', 'end_date'));
    }

    /**
     * Trial Balance
     *
     * @return Response
     */
    public function balanceSheet()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        if (! empty(request()->start_date) && ! empty(request()->end_date)) {
            $start_date = request()->start_date;
            $end_date = request()->end_date;
        } else {
            $fy = $this->businessUtil->getCurrentFinancialYear($business_id);
            $start_date = $fy['start'];
            $end_date = $fy['end'];
        }

        $balance_formula = $this->accountingUtil->balanceFormula();

        $assets = AccountingAccount::join('accounting_accounts_transactions as AAT',
                                'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                    ->join('accounting_account_types as AATP',
                                'AATP.id', '=', 'accounting_accounts.account_sub_type_id')
                    ->whereDate('AAT.operation_date', '>=', $start_date)
                    ->whereDate('AAT.operation_date', '<=', $end_date)
                    ->select(DB::raw($balance_formula), 'accounting_accounts.name', 'AATP.name as sub_type')
                    ->where('accounting_accounts.business_id', $business_id)
                    ->whereIn('accounting_accounts.account_primary_type', ['asset'])
                    ->groupBy('accounting_accounts.name')
                    ->get();

        $liabilities = AccountingAccount::join('accounting_accounts_transactions as AAT',
                                'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                    ->join('accounting_account_types as AATP',
                                'AATP.id', '=', 'accounting_accounts.account_sub_type_id')
                    ->whereDate('AAT.operation_date', '>=', $start_date)
                    ->whereDate('AAT.operation_date', '<=', $end_date)
                    ->select(DB::raw($balance_formula), 'accounting_accounts.name', 'AATP.name as sub_type')
                    ->where('accounting_accounts.business_id', $business_id)
                    ->whereIn('accounting_accounts.account_primary_type', ['liability'])
                    ->groupBy('accounting_accounts.name')
                    ->get();

        $equities = AccountingAccount::join('accounting_accounts_transactions as AAT',
                                'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                    ->join('accounting_account_types as AATP',
                                'AATP.id', '=', 'accounting_accounts.account_sub_type_id')
                    ->whereDate('AAT.operation_date', '>=', $start_date)
                    ->whereDate('AAT.operation_date', '<=', $end_date)
                    ->select(DB::raw($balance_formula), 'accounting_accounts.name', 'AATP.name as sub_type')
                    ->where('accounting_accounts.business_id', $business_id)
                    ->whereIn('accounting_accounts.account_primary_type', ['equity'])
                    ->groupBy('accounting_accounts.name')
                    ->get();

        return view('accounting::report.balance_sheet')
            ->with(compact('assets', 'liabilities', 'equities', 'start_date', 'end_date'));
    }

    public function accountReceivableAgeingReport()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        $location_id = request()->input('location_id', null);

        $report_details = $this->accountingUtil->getAgeingReport($business_id, 'sell', 'contact', $location_id);

        $business_locations = BusinessLocation::forDropdown($business_id, true);

        return view('accounting::report.account_receivable_ageing_report')
        ->with(compact('report_details', 'business_locations'));
    }

    public function accountPayableAgeingReport()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        $location_id = request()->input('location_id', null);
        $report_details = $this->accountingUtil->getAgeingReport($business_id, 'purchase', 'contact',
        $location_id);
        $business_locations = BusinessLocation::forDropdown($business_id, true);

        return view('accounting::report.account_payable_ageing_report')
        ->with(compact('report_details', 'business_locations'));
    }

    public function accountReceivableAgeingDetails()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        $location_id = request()->input('location_id', null);

        $report_details = $this->accountingUtil->getAgeingReport($business_id, 'sell', 'due_date',
        $location_id);

        $business_locations = BusinessLocation::forDropdown($business_id, true);

        return view('accounting::report.account_receivable_ageing_details')
        ->with(compact('business_locations', 'report_details'));
    }

    public function accountPayableAgeingDetails()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.view_reports'))) {
            abort(403, 'Unauthorized action.');
        }

        $location_id = request()->input('location_id', null);

        $report_details = $this->accountingUtil->getAgeingReport($business_id, 'purchase', 'due_date',
        $location_id);

        $business_locations = BusinessLocation::forDropdown($business_id, true);

        return view('accounting::report.account_payable_ageing_details')
        ->with(compact('business_locations', 'report_details'));
    }
}
