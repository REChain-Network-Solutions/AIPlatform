<?php

namespace Modules\Accounting\Http\Controllers;

use App\Utils\ModuleUtil;
use DB;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Illuminate\Routing\Controller;
use Modules\Accounting\Entities\AccountingAccount;
use Modules\Accounting\Entities\AccountingAccountsTransaction;
use Modules\Accounting\Entities\AccountingAccountType;
use Modules\Accounting\Utils\AccountingUtil;
use Yajra\DataTables\Facades\DataTables;
use Illuminate\Support\Str;

class CoaController extends Controller
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
     * Display a listing of the resource.
     *
     * @return Response
     */
    public function index()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        $account_types = AccountingAccountType::accounting_primary_type();

        foreach ($account_types as $k => $v) {
            $account_types[$k] = $v['label'];
        }

        if (request()->ajax()) {
            $balance_formula = $this->accountingUtil->balanceFormula('AA');

            $query = AccountingAccount::where('business_id', $business_id)
                                ->whereNull('parent_account_id')
                                ->with(['child_accounts' => function ($query) use ($balance_formula) {
                                    $query->select([DB::raw("(SELECT $balance_formula from accounting_accounts_transactions AS AAT
                                        JOIN accounting_accounts AS AA ON AAT.accounting_account_id = AA.id
                                        WHERE AAT.accounting_account_id = accounting_accounts.id) AS balance"), 'accounting_accounts.*']);
                                },
                                    'child_accounts.detail_type', 'detail_type', 'account_sub_type',
                                    'child_accounts.account_sub_type', ])
                                ->select([DB::raw("(SELECT $balance_formula
                                    FROM accounting_accounts_transactions AS AAT 
                                    JOIN accounting_accounts AS AA ON AAT.accounting_account_id = AA.id
                                    WHERE AAT.accounting_account_id = accounting_accounts.id) AS balance"),
                                    'accounting_accounts.*', ]);

            if (! empty(request()->input('account_type'))) {
                $query->where('accounting_accounts.account_primary_type', request()->input('account_type'));
            }
            if (! empty(request()->input('status'))) {
                $query->where('accounting_accounts.status', request()->input('status'));
            }

            $accounts = $query->get();

            $account_exist = AccountingAccount::where('business_id', $business_id)->exists();

            if (request()->input('view_type') == 'table') {
                return view('accounting::chart_of_accounts.accounts_table')
                        ->with(compact('accounts', 'account_exist'));
            } else {
                $account_sub_types = AccountingAccountType::where('account_type', 'sub_type')
                                            ->where(function ($q) use ($business_id) {
                                                $q->whereNull('business_id')
                                                    ->orWhere('business_id', $business_id);
                                            })
                                            ->get();

                return view('accounting::chart_of_accounts.accounts_tree')
                ->with(compact('accounts', 'account_exist', 'account_types', 'account_sub_types'));
            }
        }

        return view('accounting::chart_of_accounts.index')->with(compact('account_types'));
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return Response
     */
    public function create()
    {
        $business_id = request()->session()->get('user.business_id');
        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $account_types = AccountingAccountType::accounting_primary_type();

            return view('accounting::chart_of_accounts.create')->with(compact('account_types'));
        }
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return Response
     */
    public function createDefaultAccounts()
    {
        //check no accounts
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        $user_id = request()->session()->get('user.id');

        $default_accounts = [
            0 => [
                'name' => 'Accounts Payable (A/P)',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 6,
                'detail_type_id' => 58,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            1 => [
                'name' => 'Credit Card',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 7,
                'detail_type_id' => 59,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            2 => [
                'name' => 'Wage expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 140,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            3 => [
                'name' => 'Utilities',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 149,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            4 => [
                'name' => 'Unrealised loss on securities, net of tax',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 12,
                'detail_type_id' => 113,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            5 => [
                'name' => 'Undeposited Funds',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 29,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            6 => [
                'name' => 'Uncategorised Income',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => 103,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            7 => [
                'name' => 'Uncategorised Expense',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 138,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            8 => [
                'name' => 'Uncategorised Asset',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 26,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            9 => [
                'name' => 'Unapplied Cash Payment Income',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => 105,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            10 => [
                'name' => 'Travel expenses - selling expense',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => '147',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            11 => [
                'name' => 'Travel expenses - general and admin expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => '146',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            12 => [
                'name' => 'Supplies',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 145,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            13 => [
                'name' => 'Subcontractors - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            14 => [
                'name' => 'Stationery and printing',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => '137',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            15 => [
                'name' => 'Short-term debit',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 8,
                'detail_type_id' => 69,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            16 => [
                'name' => 'Shipping and delivery expense',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 143,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            17 => [
                'name' => 'Share capital',
                'business_id' => $business_id,
                'account_primary_type' => 'equity',
                'account_sub_type_id' => 10,
                'detail_type_id' => 95,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            18 => [
                'name' => 'Sales of Product Income',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => 103,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            19 => [
                'name' => 'Sales - wholesale',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => '102',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            20 => [
                'name' => 'Sales - retail',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => '101',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            21 => [
                'name' => 'Sales',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => 103,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            22 => [
                'name' => 'Revenue - General',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => '100',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            23 => [
                'name' => 'Retained Earnings',
                'business_id' => $business_id,
                'account_primary_type' => 'equity',
                'account_sub_type_id' => 10,
                'detail_type_id' => 94,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            24 => [
                'name' => 'Repair and maintenance',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 142,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            25 => [
                'name' => 'Rent or lease payments',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 141,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            26 => [
                'name' => 'Reconciliation Discrepancies',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 15,
                'detail_type_id' => 153,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            27 => [
                'name' => 'Purchases',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 144,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            28 => [
                'name' => 'Property, plant and equipment',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 4,
                'detail_type_id' => 42,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            29 => [
                'name' => 'Prepaid Expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 27,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            30 => [
                'name' => 'Payroll liabilities',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 8,
                'detail_type_id' => 71,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            31 => [
                'name' => 'Payroll Expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 140,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            32 => [
                'name' => 'Payroll Clearing',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 8,
                'detail_type_id' => 70,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            33 => [
                'name' => 'Overhead - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            34 => [
                'name' => 'Other Types of Expenses-Advertising Expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => '119',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            35 => [
                'name' => 'Other selling expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 139,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            36 => [
                'name' => 'Other operating income (expenses)',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 12,
                'detail_type_id' => 111,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            37 => [
                'name' => 'Other general and administrative expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => '137',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            38 => [
                'name' => 'Other comprehensive income',
                'business_id' => $business_id,
                'account_primary_type' => 'equity',
                'account_sub_type_id' => 10,
                'detail_type_id' => 87,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            39 => [
                'name' => 'Other - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            40 => [
                'name' => 'Office expenses',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => '137',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            41 => [
                'name' => 'Meals and entertainment',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 137,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            42 => [
                'name' => 'Materials - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            43 => [
                'name' => 'Management compensation',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 135,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            44 => [
                'name' => 'Loss on disposal of assets',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 12,
                'detail_type_id' => 108,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            45 => [
                'name' => 'Loss on discontinued operations, net of tax',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 134,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            46 => [
                'name' => 'Long-term investments',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 5,
                'detail_type_id' => 54,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            47 => [
                'name' => 'Long-term debt',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 9,
                'detail_type_id' => 78,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            48 => [
                'name' => 'Liabilities related to assets held for sale',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 9,
                'detail_type_id' => 77,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            49 => [
                'name' => 'Legal and professional fees',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 133,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            50 => [
                'name' => 'Inventory Asset',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 21,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            51 => [
                'name' => 'Inventory',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 21,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            52 => [
                'name' => 'Interest income',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 12,
                'detail_type_id' => 107,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            53 => [
                'name' => 'Interest expense',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 132,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            54 => [
                'name' => 'Intangibles',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 5,
                'detail_type_id' => 51,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            55 => [
                'name' => 'Insurance - Liability',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 131,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            56 => [
                'name' => 'Insurance - General',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 131,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            57 => [
                'name' => 'Insurance - Disability',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 131,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            58 => [
                'name' => 'Income tax payable',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 8,
                'detail_type_id' => 65,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            59 => [
                'name' => 'Income tax expense',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 130,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            60 => [
                'name' => 'Goodwill',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 5,
                'detail_type_id' => 50,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            61 => [
                'name' => 'Freight and delivery - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            62 => [
                'name' => 'Equity in earnings of subsidiaries',
                'business_id' => $business_id,
                'account_primary_type' => 'equity',
                'account_sub_type_id' => 10,
                'detail_type_id' => 84,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            63 => [
                'name' => 'Equipment rental',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 128,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            64 => [
                'name' => 'Dues and Subscriptions',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 127,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            65 => [
                'name' => 'Dividends payable',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 8,
                'detail_type_id' => 64,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            66 => [
                'name' => 'Dividend income',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 12,
                'detail_type_id' => 106,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            67 => [
                'name' => 'Dividend disbursed',
                'business_id' => $business_id,
                'account_primary_type' => 'equity',
                'account_sub_type_id' => 10,
                'detail_type_id' => 83,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            68 => [
                'name' => 'Discounts given - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            69 => [
                'name' => 'Direct labour - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            70 => [
                'name' => 'Deferred tax assets',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 5,
                'detail_type_id' => 49,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            71 => [
                'name' => 'Cost of sales',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '118',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            72 => [
                'name' => 'Commissions and fees',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 125,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            73 => [
                'name' => 'Change in inventory - COS',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 13,
                'detail_type_id' => '114',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            74 => [
                'name' => 'Cash and cash equivalents',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 3,
                'detail_type_id' => 31,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            75 => [
                'name' => 'Billable Expense Income',
                'business_id' => $business_id,
                'account_primary_type' => 'income',
                'account_sub_type_id' => 11,
                'detail_type_id' => 103,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            76 => [
                'name' => 'Bank charges',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 123,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            77 => [
                'name' => 'Bad debts',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 122,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            78 => [
                'name' => 'Available for sale assets (short-term)',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 18,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            79 => [
                'name' => 'Assets held for sale',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 5,
                'detail_type_id' => 48,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            80 => [
                'name' => 'Amortisation expense',
                'business_id' => $business_id,
                'account_primary_type' => 'expenses',
                'account_sub_type_id' => 14,
                'detail_type_id' => 120,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            81 => [
                'name' => 'Allowance for bad debts',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 2,
                'detail_type_id' => 17,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            82 => [
                'name' => 'Accumulated depreciation on property, plant and equipment',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 4,
                'detail_type_id' => 38,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            83 => [
                'name' => 'Accrued non-current liabilities',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 9,
                'detail_type_id' => 76,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            84 => [
                'name' => 'Accrued liabilities',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 8,
                'detail_type_id' => 60,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            85 => [
                'name' => 'Accrued holiday payable',
                'business_id' => $business_id,
                'account_primary_type' => 'liability',
                'account_sub_type_id' => 9,
                'detail_type_id' => 75,
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
            86 => [
                'name' => 'Accounts Receivable (A/R)',
                'business_id' => $business_id,
                'account_primary_type' => 'asset',
                'account_sub_type_id' => 1,
                'detail_type_id' => '16',
                'status' => 'active',
                'created_by' => $user_id,
                'created_at' => \Carbon::now(),
                'updated_at' => \Carbon::now(),
            ],
        ];

        if (AccountingAccount::where('business_id', $business_id)->doesntExist()) {
            AccountingAccount::insert($default_accounts);
        }

        //redirect back
        $output = ['success' => 1,
            'msg' => __('lang_v1.added_success'),
        ];

        return redirect()->back()->with('status', $output);
    }

    public function getAccountDetailsType()
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $account_type_id = request()->input('account_type_id');
            $detail_types_obj = AccountingAccountType::where('parent_id', $account_type_id)
                                    ->where(function ($q) use ($business_id) {
                                        $q->whereNull('business_id')
                                            ->orWhere('business_id', $business_id);
                                    })
                                    ->where('account_type', 'detail_type')
                                    ->get();

            $parent_accounts = AccountingAccount::where('business_id', $business_id)
                                            ->where('account_sub_type_id', $account_type_id)
                                            ->whereNull('parent_account_id')
                                            ->select('name as text', 'id')
                                            ->get();
            $parent_accounts->prepend([
                'id' => 'null',
                'text' => __('messages.please_select'),
            ]);

            $detail_types = [[
                'id' => 'null',
                'text' => __('messages.please_select'),
                'description' => '',
            ]];

            foreach ($detail_types_obj as $detail_type) {
                $detail_types[] = [
                    'id' => $detail_type->id,
                    'text' => __('accounting::lang.'.$detail_type->name),
                    'description' => ! empty($detail_type->description) ?
                        __('accounting::lang.'.$detail_type->description) : '',
                ];
            }

            return [
                'detail_types' => $detail_types,
                'parent_accounts' => $parent_accounts,
            ];
        }
    }

    public function getAccountSubTypes()
    {
        if (request()->ajax()) {
            $business_id = request()->session()->get('user.business_id');
            $account_primary_type = request()->input('account_primary_type');
            $sub_types_obj = AccountingAccountType::where('account_primary_type', $account_primary_type)
                                        ->where(function ($q) use ($business_id) {
                                            $q->whereNull('business_id')
                                                ->orWhere('business_id', $business_id);
                                        })
                                        ->where('account_type', 'sub_type')
                                        ->get();

            $sub_types = [[
                'id' => 'null',
                'text' => __('messages.please_select'),
                'show_balance' => 0,
            ]];

            foreach ($sub_types_obj as $st) {
                $sub_types[] = [
                    'id' => $st->id,
                    'text' => $st->account_type_name,
                    'show_balance' => $st->show_balance,
                ];
            }

            return [
                'sub_types' => $sub_types,
            ];
        }
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  Request  $request
     * @return Response
     */
    public function store(Request $request)
    {
        $business_id = $request->session()->get('user.business_id');
        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        try {
            DB::beginTransaction();

            $input = $request->only(['name', 'account_primary_type', 'account_sub_type_id', 'detail_type_id',
                'parent_account_id', 'description', 'gl_code', ]);

            $account_type = AccountingAccountType::find($input['account_sub_type_id']);

            $input['parent_account_id'] = ! empty($input['parent_account_id'])
            && $input['parent_account_id'] !== 'null' ? $input['parent_account_id'] : null;
            $input['created_by'] = auth()->user()->id;
            $input['business_id'] = $request->session()->get('user.business_id');
            $input['status'] = 'active';

            $account = AccountingAccount::create($input);

            if ($account_type->show_balance == 1 && ! empty($request->input('balance'))) {
                //Opening balance
                $data = [
                    'amount' => $this->accountingUtil->num_uf($request->input('balance')),
                    'accounting_account_id' => $account->id,
                    'created_by' => auth()->user()->id,
                    'operation_date' => ! empty($request->input('balance_as_of')) ?
                    $this->accountingUtil->uf_date($request->input('balance_as_of')) :
                    \Carbon::today()->format('Y-m-d'),
                ];

                //Opening balance
                $data['type'] = in_array($input['account_primary_type'], ['asset', 'expenses']) ? 'debit' : 'credit';
                $data['sub_type'] = 'opening_balance';
                AccountingAccountsTransaction::createTransaction($data);
            }

            DB::commit();
        } catch (\Exception $e) {
            DB::rollBack();

            \Log::emergency('File:'.$e->getFile().'Line:'.$e->getLine().'Message:'.$e->getMessage());
        }

        return redirect()->back();
    }

    /**
     * Show the specified resource.
     *
     * @param  int  $id
     * @return Response
     */
    public function show($id)
    {
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  int  $id
     * @return Response
     */
    public function edit($id)
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $account = AccountingAccount::where('business_id', $business_id)
                                    ->with(['detail_type'])
                                    ->find($id);

            $account_types = AccountingAccountType::accounting_primary_type();
            $account_sub_types = AccountingAccountType::where('account_primary_type', $account->account_primary_type)
                                            ->where('account_type', 'sub_type')
                                            ->where(function ($q) use ($business_id) {
                                                $q->whereNull('business_id')
                                                    ->orWhere('business_id', $business_id);
                                            })
                                            ->get();
            $account_detail_types = AccountingAccountType::where('parent_id', $account->account_sub_type_id)
                                    ->where('account_type', 'detail_type')
                                    ->where(function ($q) use ($business_id) {
                                        $q->whereNull('business_id')
                                            ->orWhere('business_id', $business_id);
                                    })
                                    ->get();

            $parent_accounts = AccountingAccount::where('business_id', $business_id)
                                    ->where('account_sub_type_id', $account->account_sub_type_id)
                                    ->whereNull('parent_account_id')
                                    ->get();

            return view('accounting::chart_of_accounts.edit')->with(compact('account_types', 'account',
                                                'account_sub_types', 'account_detail_types', 'parent_accounts'));
        }
    }

    /**
     * Update the specified resource in storage.
     *
     * @param  Request  $request
     * @param  int  $id
     * @return Response
     */
    public function update(Request $request, $id)
    {
        $business_id = $request->session()->get('user.business_id');
        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        try {
            DB::beginTransaction();

            $input = $request->only(['name', 'account_primary_type', 'account_sub_type_id', 'detail_type_id',
                'parent_account_id', 'description', 'gl_code', ]);

            $input['parent_account_id'] = ! empty($input['parent_account_id'])
            && $input['parent_account_id'] !== 'null' ? $input['parent_account_id'] : null;

            $account = AccountingAccount::find($id);
            $account->update($input);

            DB::commit();
        } catch (\Exception $e) {
            DB::rollBack();

            \Log::emergency('File:'.$e->getFile().'Line:'.$e->getLine().'Message:'.$e->getMessage());
        }

        return redirect()->back();
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  int  $id
     * @return Response
     */
    public function destroy($id)
    {
        //
    }

    public function activateDeactivate($id)
    {
        $business_id = request()->session()->get('user.business_id');
        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $account = AccountingAccount::where('business_id', $business_id)
                                    ->find($id);

            $account->status = $account->status == 'active' ? 'inactive' : 'active';
            $account->save();

            $msg = $account->status == 'active' ? __('accounting::lang.activated_successfully') :
            __('accounting::lang.deactivated_successfully');
            $output = ['success' => 1,
                'msg' => $msg,
            ];

            return $output;
        }
    }

    /**
     * Displays the ledger of the account
     *
     * @param  int  $account_id
     * @return Response
     */
    public function ledger($account_id)
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_accounts'))) {
            abort(403, 'Unauthorized action.');
        }

        $account = AccountingAccount::where('business_id', $business_id)
                        ->with(['account_sub_type', 'detail_type'])
                        ->findorFail($account_id);

        if (request()->ajax()) {
            $start_date = request()->input('start_date');
            $end_date = request()->input('end_date');

            // $before_bal_query = AccountingAccountsTransaction::where('accounting_account_id', $account->id)
            //                     ->leftjoin('accounting_acc_trans_mappings as ATM', 'accounting_accounts_transactions.acc_trans_mapping_id', '=', 'ATM.id')
            //         ->select([
            //             DB::raw('SUM(IF(accounting_accounts_transactions.type="credit", accounting_accounts_transactions.amount, -1 * accounting_accounts_transactions.amount)) as prev_bal')])
            //         ->where('accounting_accounts_transactions.operation_date', '<', $start_date);
            // $bal_before_start_date = $before_bal_query->first()->prev_bal;

            $transactions = AccountingAccountsTransaction::where('accounting_account_id', $account->id)
                            ->leftjoin('accounting_acc_trans_mappings as ATM', 'accounting_accounts_transactions.acc_trans_mapping_id', '=', 'ATM.id')
                            ->leftjoin('transactions as T', 'accounting_accounts_transactions.transaction_id', '=', 'T.id')
                            ->leftjoin('users AS U', 'accounting_accounts_transactions.created_by', 'U.id')
                            ->select('accounting_accounts_transactions.operation_date',
                                'accounting_accounts_transactions.sub_type',
                                'accounting_accounts_transactions.type',
                                'accounting_accounts_transactions.note as aat_note',
                                'ATM.ref_no as a_ref', 'ATM.note',
                                'accounting_accounts_transactions.amount',
                                DB::raw("CONCAT(COALESCE(U.surname, ''),' ',COALESCE(U.first_name, ''),' ',COALESCE(U.last_name,'')) as added_by"),
                                'T.invoice_no', 'T.ref_no'
                            );
            if (! empty($start_date) && ! empty($end_date)) {
                $transactions->whereDate('accounting_accounts_transactions.operation_date', '>=', $start_date)
                        ->whereDate('accounting_accounts_transactions.operation_date', '<=', $end_date);
            }

            return DataTables::of($transactions)
                    ->editColumn('operation_date', function ($row) {
                        return $this->accountingUtil->format_date($row->operation_date, true);
                    })
                    ->editColumn('ref_no', function ($row) {
                        $description = '';

                        if ($row->sub_type == 'journal_entry') {
                            $description = '<b>'.__('accounting::lang.journal_entry').'</b>';
                            $description .= '<br>'.__('purchase.ref_no').': '.$row->a_ref;
                            $description .= '<br>'.__('lang_v1.description').': '.$row->aat_note;
                        }

                        if ($row->sub_type == 'opening_balance') {
                            $description = '<b>'.__('accounting::lang.opening_balance').'</b>';
                            $description .= '<br>'.__('lang_v1.description').': '.$row->aat_note;
                        }

                        if ($row->sub_type == 'sell') {
                            $description = '<b>'.__('sale.sale').'</b>';
                            $description .= '<br>'.__('sale.invoice_no').': '.$row->invoice_no;
                            $description .= '<br>'.__('lang_v1.description').': '.$row->aat_note;
                        }

                        if ($row->sub_type == 'expense') {
                            $description = '<b>'.__('accounting::lang.expense').'</b>';
                            $description .= '<br>'.__('purchase.ref_no').': '.$row->ref_no;
                            $description .= '<br>'.__('lang_v1.description').': '.$row->aat_note;
                        }

                        return $description;
                    })
                    ->addColumn('debit', function ($row) {
                        if ($row->type == 'debit') {
                            return '<span class="debit" data-orig-value="'.$row->amount.'">'.$this->accountingUtil->num_f($row->amount, true).'</span>';
                        }

                        return '';
                    })
                    ->addColumn('credit', function ($row) {
                        if ($row->type == 'credit') {
                            return '<span class="credit"  data-orig-value="'.$row->amount.'">'.$this->accountingUtil->num_f($row->amount, true).'</span>';
                        }

                        return '';
                    })
                    // ->addColumn('balance', function ($row) use ($bal_before_start_date, $start_date) {
                    //     //TODO:: Need to fix same balance showing for transactions having same operation date
                    //     $current_bal = AccountingAccountsTransaction::where('accounting_account_id',
                    //                         $row->account_id)
                    //                     ->where('operation_date', '>=', $start_date)
                    //                     ->where('operation_date', '<=', $row->operation_date)
                    //                     ->select(DB::raw("SUM(IF(type='credit', amount, -1 * amount)) as balance"))
                    //                     ->first()->balance;
                    //     $bal = $bal_before_start_date + $current_bal;
                    //     return '<span class="balance" data-orig-value="' . $bal . '">' . $this->accountingUtil->num_f($bal, true) . '</span>';
                    // })
                    ->editColumn('action', function ($row) {
                        $action = '';

                        return $action;
                    })
                    ->filterColumn('added_by', function ($query, $keyword) {
                        $query->whereRaw("CONCAT(COALESCE(u.surname, ''), ' ', COALESCE(u.first_name, ''), ' ', COALESCE(u.last_name, '')) like ?", ["%{$keyword}%"]);
                    })
                    ->rawColumns(['ref_no', 'credit', 'debit', 'balance', 'action'])
                    ->make(true);
        }

        $current_bal = AccountingAccount::leftjoin('accounting_accounts_transactions as AAT',
                            'AAT.accounting_account_id', '=', 'accounting_accounts.id')
                        ->where('business_id', $business_id)
                        ->where('accounting_accounts.id', $account->id)
                        ->select([DB::raw($this->accountingUtil->balanceFormula())]);
        $current_bal = $current_bal->first()->balance;

        return view('accounting::chart_of_accounts.ledger')
            ->with(compact('account', 'current_bal'));
    }
}
