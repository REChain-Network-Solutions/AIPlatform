<?php

namespace Modules\Accounting\Http\Controllers;

use App\Http\Controllers\Controller;
use App\Utils\ModuleUtil;
use DB;
use Excel;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Modules\Accounting\Entities\AccountingAccount;
use Modules\Accounting\Entities\AccountingAccountType;
use Modules\Accounting\Entities\AccountingBudget;
use Modules\Accounting\Exports\BudgetExport;

class BudgetController extends Controller
{
    protected $moduleUtil;

    /**
     * Constructor
     *
     * @return void
     */
    public function __construct(ModuleUtil $moduleUtil)
    {
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
            ! (auth()->user()->can('accounting.manage_budget'))) {
            abort(403, 'Unauthorized action.');
        }

        $fy_year = request()->input('financial_year', null);
        $budget = [];
        $accounts = [];
        if ($fy_year != null) {
            $accounts = AccountingAccount::where('business_id', $business_id)
                                ->where('status', 'active')
                                ->select('name', 'id', 'account_primary_type')
                                ->get();

            $budget = AccountingBudget::whereIn('accounting_account_id', $accounts->pluck('id'))
                                ->where('financial_year', $fy_year)
                                ->get();

            if (count($budget) == 0) {
                return redirect(action([\Modules\Accounting\Http\Controllers\BudgetController::class, 'create']).
                '?financial_year='.$fy_year);
            }
        }
        $months = $this->getFinancialYearMonths();

        $account_types = AccountingAccountType::accounting_primary_type();

        if (! empty(request()->input('format'))) {
            $view_type = request()->input('view_type');
            if (request()->input('view_type') == 'monthly') {
                if (request()->input('format') == 'pdf') {
                    $html = view('accounting::budget.partials.budget_monthly_pdf')->with(compact('accounts',
                    'budget', 'months', 'fy_year', 'account_types'))->render();

                    $output_file_name = 'Budget-'.$fy_year.'-Monthly.pdf';

                    $mpdf = $this->getMpdf();
                    $mpdf->WriteHTML($html);
                    $mpdf->Output($output_file_name, 'D');
                } elseif (request()->input('format') == 'excel') {
                    $export = new BudgetExport($accounts, $budget, $months, $fy_year, $account_types, $view_type);

                    $output_file_name = 'Budget-'.$fy_year.'-Monthly.xlsx';

                    return Excel::download($export, $output_file_name);
                } elseif (request()->input('format') == 'csv') {
                    $export = new BudgetExport($accounts, $budget, $months, $fy_year, $account_types, $view_type);

                    $output_file_name = 'Budget-'.$fy_year.'-Monthly.csv';

                    return Excel::download($export, $output_file_name);
                }
            } elseif (request()->input('view_type') == 'quarterly') {
                if (request()->input('format') == 'pdf') {
                    $html = view('accounting::budget.partials.budget_quarterly_pdf')->with(compact('accounts',
                    'budget', 'fy_year', 'account_types'))->render();

                    $output_file_name = 'Budget-'.$fy_year.'-Quarterly.pdf';

                    $mpdf = $this->getMpdf();
                    $mpdf->WriteHTML($html);
                    $mpdf->Output($output_file_name, 'D');
                } elseif (request()->input('format') == 'excel') {
                    $export = new BudgetExport($accounts, $budget, $months, $fy_year, $account_types, $view_type);

                    $output_file_name = 'Budget-'.$fy_year.'-Quarterly.xlsx';

                    return Excel::download($export, $output_file_name);
                } elseif (request()->input('format') == 'csv') {
                    $export = new BudgetExport($accounts, $budget, $months, $fy_year, $account_types, $view_type);

                    $output_file_name = 'Budget-'.$fy_year.'-Quarterly.csv';

                    return Excel::download($export, $output_file_name);
                }
            } elseif (request()->input('view_type') == 'yearly') {
                if (request()->input('format') == 'pdf') {
                    $html = view('accounting::budget.partials.budget_yearly_pdf')->with(compact('accounts',
                    'budget', 'fy_year', 'account_types'))->render();

                    $output_file_name = 'Budget-'.$fy_year.'-Yearly.pdf';

                    $mpdf = $this->getMpdf();
                    $mpdf->WriteHTML($html);
                    $mpdf->Output($output_file_name, 'D');
                } elseif (request()->input('format') == 'excel') {
                    $export = new BudgetExport($accounts, $budget, $months, $fy_year, $account_types, $view_type);

                    $output_file_name = 'Budget-'.$fy_year.'-Yearly.xlsx';

                    return Excel::download($export, $output_file_name);
                } elseif (request()->input('format') == 'csv') {
                    $export = new BudgetExport($accounts, $budget, $months, $fy_year, $account_types, $view_type);

                    $output_file_name = 'Budget-'.$fy_year.'-Yearly.csv';

                    return Excel::download($export, $output_file_name);
                }
            }
        }

        return view('accounting::budget.index')->with(compact('accounts', 'budget', 'months', 'fy_year',
                'account_types'));
    }

    private function getFinancialYearMonths()
    {
        $fy_start_month = request()->session()->get('business.fy_start_month');
        $months = [];
        $month_names = [
            1 => 'jan',
            2 => 'feb',
            3 => 'mar',
            4 => 'apr',
            5 => 'may',
            6 => 'jun',
            7 => 'jul',
            8 => 'aug',
            9 => 'sep',
            10 => 'oct',
            11 => 'nov',
            12 => 'dec',
        ];
        for ($i = $fy_start_month; count($months) <= 11; $i++) {
            $months[$i] = $month_names[$i];
            if ($i == 12) {
                $i = 0;
            }
        }

        return $months;
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
            ! (auth()->user()->can('accounting.manage_budget'))) {
            abort(403, 'Unauthorized action.');
        }

        $fy_year = request()->input('financial_year');

        $accounts = AccountingAccount::where('business_id', $business_id)
                                ->where('status', 'active')
                                ->select('name', 'id')
                                ->get();
        $months = $this->getFinancialYearMonths();

        $budget = AccountingBudget::whereIn('accounting_account_id', $accounts->pluck('id'))
                                ->where('financial_year', $fy_year)
                                ->get();

        return view('accounting::budget.create')->with(compact('fy_year', 'accounts', 'months', 'budget'));
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  Request  $request
     * @return Response
     */
    public function store(Request $request)
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.manage_budget'))) {
            abort(403, 'Unauthorized action.');
        }

        try {
            DB::beginTransaction();

            foreach ($request->input('budget') as $key => $value) {
                $inputs = [];
                foreach ($value as $k => $v) {
                    $inputs[$k] = ! is_null($v) ? $this->moduleUtil->num_uf($v) : null;
                }
                AccountingBudget::updateOrCreate(
                    [
                        'accounting_account_id' => $key,
                        'financial_year' => $request->input('financial_year'),
                    ],
                    $inputs
                );
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
        return view('accounting::show');
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  int  $id
     * @return Response
     */
    public function edit($id)
    {
        return view('accounting::edit');
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
        //
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
}
