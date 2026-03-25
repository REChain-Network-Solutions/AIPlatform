<?php

namespace Modules\Accounting\Http\Controllers;

use App\Utils\ModuleUtil;
use App\Utils\Util;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\DB;
use Modules\Accounting\Entities\AccountingAccountsTransaction;
use Modules\Accounting\Entities\AccountingAccTransMapping;
use Modules\Accounting\Utils\AccountingUtil;
use Yajra\DataTables\Facades\DataTables;

class TransferController extends Controller
{
    /**
     * All Utils instance.
     */
    protected $util;

    /**
     * Constructor
     *
     * @param  ProductUtils  $product
     * @return void
     */
    public function __construct(Util $util, ModuleUtil $moduleUtil, AccountingUtil $accountingUtil)
    {
        $this->util = $util;
        $this->moduleUtil = $moduleUtil;
        $this->accountingUtil = $accountingUtil;
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
            ! (auth()->user()->can('accounting.view_transfer'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $transfers = AccountingAccTransMapping::where('accounting_acc_trans_mappings.business_id', $business_id)
                        ->join('users as u', 'accounting_acc_trans_mappings.created_by', 'u.id')
                        ->join('accounting_accounts_transactions as from_transaction', function ($join) {
                            $join->on('from_transaction.acc_trans_mapping_id', '=', 'accounting_acc_trans_mappings.id')
                                    ->where('from_transaction.type', 'debit');
                        })
                        ->join('accounting_accounts_transactions as to_transaction', function ($join) {
                            $join->on('to_transaction.acc_trans_mapping_id', '=', 'accounting_acc_trans_mappings.id')
                                    ->where('to_transaction.type', 'credit');
                        })
                        ->join('accounting_accounts as from_account',
                        'from_transaction.accounting_account_id', 'from_account.id')
                        ->join('accounting_accounts as to_account',
                        'to_transaction.accounting_account_id', 'to_account.id')
                        ->where('accounting_acc_trans_mappings.type', 'transfer')
                        ->select(['accounting_acc_trans_mappings.id',
                            'accounting_acc_trans_mappings.ref_no',
                            'accounting_acc_trans_mappings.operation_date',
                            'accounting_acc_trans_mappings.note',
                            DB::raw("CONCAT(COALESCE(u.surname, ''),' ',COALESCE(u.first_name, ''),' ',COALESCE(u.last_name,'')) 
                            as added_by"),
                            'from_transaction.amount',
                            'from_account.name as from_account_name',
                            'to_account.name as to_account_name',
                        ]);

            if (! empty(request()->start_date) && ! empty(request()->end_date)) {
                $start = request()->start_date;
                $end = request()->end_date;
                $transfers->whereDate('accounting_acc_trans_mappings.operation_date', '>=', $start)
                            ->whereDate('accounting_acc_trans_mappings.operation_date', '<=', $end);
            }

            if (! empty(request()->transfer_from)) {
                $transfers->where('from_account.id', request()->transfer_from);
            }

            if (! empty(request()->transfer_to)) {
                $transfers->where('to_account.id', request()->transfer_to);
            }

            return Datatables::of($transfers)
                ->addColumn(
                    'action', function ($row) {
                        $html = '<div class="btn-group">
                                <button type="button" class="tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline tw-dw-btn-info tw-w-max dropdown-toggle" 
                                    data-toggle="dropdown" aria-expanded="false">'.
                                    __('messages.actions').
                                    '<span class="caret"></span><span class="sr-only">Toggle Dropdown
                                    </span>
                                </button>
                                <ul class="dropdown-menu dropdown-menu-right" role="menu">';
                        if (auth()->user()->can('accounting.edit_transfer')) {
                            $html .= '<li>
                                <a href="#" data-href="'.action([\Modules\Accounting\Http\Controllers\TransferController::class, 'edit'],
                                [$row->id]).'" class="btn-modal" data-container="#create_transfer_modal">
                                    <i class="fas fa-edit"></i>'.__('messages.edit').'
                                </a>
                            </li>';
                        }
                        if (auth()->user()->can('accounting.delete_transfer')) {
                            $html .= '<li>
                                    <a href="#" data-href="'.action([\Modules\Accounting\Http\Controllers\TransferController::class, 'destroy'], [$row->id]).'" class="delete_transfer_button">
                                        <i class="fas fa-trash" aria-hidden="true"></i>'.__('messages.delete').'
                                    </a>
                                    </li>';
                        }

                        $html .= '</ul></div>';

                        return $html;
                    })
                ->editColumn('amount', function ($row) {
                    return $this->util->num_f($row->amount, true);
                })
                ->editColumn('operation_date', function ($row) {
                    return $this->util->format_date($row->operation_date, true);
                })
                ->filterColumn('added_by', function ($query, $keyword) {
                    $query->whereRaw("CONCAT(COALESCE(u.surname, ''), ' ', 
                    COALESCE(u.first_name, ''), ' ', COALESCE(u.last_name, '')) like ?", ["%{$keyword}%"]);
                })
                ->rawColumns(['action'])
                ->make(true);
        }

        return view('accounting::transfer.index');
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
            ! (auth()->user()->can('accounting.add_transfer'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            return view('accounting::transfer.create');
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
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.add_transfer'))) {
            abort(403, 'Unauthorized action.');
        }

        try {
            DB::beginTransaction();

            $user_id = request()->session()->get('user.id');

            $from_account = $request->get('from_account');
            $to_account = $request->get('to_account');
            $amount = $request->get('amount');
            $date = $this->util->uf_date($request->get('operation_date'), true);

            $accounting_settings = $this->accountingUtil->getAccountingSettings($business_id);

            $ref_no = $request->get('ref_no');
            $ref_count = $this->util->setAndGetReferenceCount('accounting_transfer');
            if (empty($ref_no)) {
                $prefix = ! empty($accounting_settings['transfer_prefix']) ?
                $accounting_settings['transfer_prefix'] : '';

                //Generate reference number
                $ref_no = $this->util->generateReferenceNumber('accounting_transfer', $ref_count, $business_id, $prefix);
            }

            $acc_trans_mapping = new AccountingAccTransMapping();
            $acc_trans_mapping->business_id = $business_id;
            $acc_trans_mapping->ref_no = $ref_no;
            $acc_trans_mapping->note = $request->get('note');
            $acc_trans_mapping->type = 'transfer';
            $acc_trans_mapping->created_by = $user_id;
            $acc_trans_mapping->operation_date = $date;
            $acc_trans_mapping->save();

            $from_transaction_data = [
                'acc_trans_mapping_id' => $acc_trans_mapping->id,
                'amount' => $this->util->num_uf($amount),
                'type' => 'debit',
                'sub_type' => 'transfer',
                'accounting_account_id' => $from_account,
                'created_by' => $user_id,
                'operation_date' => $date,
            ];

            $to_transaction_data = $from_transaction_data;
            $to_transaction_data['accounting_account_id'] = $to_account;
            $to_transaction_data['type'] = 'credit';

            AccountingAccountsTransaction::create($from_transaction_data);
            AccountingAccountsTransaction::create($to_transaction_data);

            DB::commit();

            $output = ['success' => 1,
                'msg' => __('lang_v1.added_success'),
            ];
        } catch (\Exception $e) {
            DB::rollBack();
            \Log::emergency('File:'.$e->getFile().'Line:'.$e->getLine().'Message:'.$e->getMessage());

            $output = ['success' => 0,
                'msg' => __('messages.something_went_wrong'),
            ];
        }

        return $output;
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
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.edit_transfer'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $mapping_transaction = AccountingAccTransMapping::where('id', $id)
                            ->where('business_id', $business_id)->firstOrFail();

            $debit_tansaction = AccountingAccountsTransaction::where('acc_trans_mapping_id', $id)
                                    ->where('type', 'debit')
                                    ->first();
            $credit_tansaction = AccountingAccountsTransaction::where('acc_trans_mapping_id', $id)
                                    ->where('type', 'credit')
                                    ->first();

            return view('accounting::transfer.edit')->with(compact('mapping_transaction',
            'debit_tansaction', 'credit_tansaction'));
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
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.edit_transfer'))) {
            abort(403, 'Unauthorized action.');
        }

        try {
            $mapping_transaction = AccountingAccTransMapping::where('id', $id)
                            ->where('business_id', $business_id)->firstOrFail();

            $debit_tansaction = AccountingAccountsTransaction::where('acc_trans_mapping_id', $id)
                                    ->where('type', 'debit')
                                    ->first();
            $credit_tansaction = AccountingAccountsTransaction::where('acc_trans_mapping_id', $id)
                                    ->where('type', 'credit')
                                    ->first();

            DB::beginTransaction();
            $from_account = $request->get('from_account');
            $to_account = $request->get('to_account');
            $amount = $request->get('amount');
            $date = $this->util->uf_date($request->get('operation_date'), true);

            $ref_no = $request->get('ref_no');
            $ref_count = $this->util->setAndGetReferenceCount('accounting_transfer');
            if (empty($ref_no)) {
                //Generate reference number
                $ref_no = $this->util->generateReferenceNumber('accounting_transfer', $ref_count);
            }

            $mapping_transaction->ref_no = $ref_no;
            $mapping_transaction->note = $request->get('note');
            $mapping_transaction->operation_date = $date;
            $mapping_transaction->save();

            $debit_tansaction->accounting_account_id = $from_account;
            $debit_tansaction->operation_date = $date;
            $debit_tansaction->amount = $this->util->num_uf($amount);
            $debit_tansaction->save();

            $credit_tansaction->accounting_account_id = $to_account;
            $credit_tansaction->operation_date = $date;
            $credit_tansaction->amount = $this->util->num_uf($amount);
            $credit_tansaction->save();

            DB::commit();

            $output = ['success' => 1,
                'msg' => __('lang_v1.updated_success'),
            ];
        } catch (\Exception $e) {
            DB::rollBack();
            \Log::emergency('File:'.$e->getFile().'Line:'.$e->getLine().'Message:'.$e->getMessage());

            $output = ['success' => 0,
                'msg' => __('messages.something_went_wrong'),
            ];
        }

        return $output;
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  int  $id
     * @return Response
     */
    public function destroy($id)
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.delete_transfer'))) {
            abort(403, 'Unauthorized action.');
        }

        $user_id = request()->session()->get('user.id');

        $acc_trans_mapping = AccountingAccTransMapping::where('id', $id)
                        ->where('business_id', $business_id)->firstOrFail();

        if (! empty($acc_trans_mapping)) {
            $acc_trans_mapping->delete();
            AccountingAccountsTransaction::where('acc_trans_mapping_id', $id)->delete();
        }

        return ['success' => 1,
            'msg' => __('lang_v1.deleted_success'),
        ];
    }
}
