<?php

namespace Modules\Accounting\Http\Controllers;

use App\BusinessLocation;
use App\Contact;
use App\Transaction;
use App\TransactionPayment;
use App\Utils\ModuleUtil;
use App\Utils\TransactionUtil;
use Modules\Accounting\Utils\AccountingUtil;
use DB;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Illuminate\Routing\Controller;
use Modules\Accounting\Entities\AccountingAccount;
use Modules\Accounting\Entities\AccountingAccountsTransaction;
use Yajra\DataTables\Facades\DataTables;

class TransactionController extends Controller
{
    /**
     * All Utils instance.
     */
    protected $transactionUtil;
    protected $moduleUtil;
    protected $accountingUtil;

    /**
     * Constructor
     *
     * @param  ProductUtils  $product
     * @return void
     */
    public function __construct(TransactionUtil $transactionUtil, ModuleUtil $moduleUtil, AccountingUtil $accountingUtil)
    {
        $this->transactionUtil = $transactionUtil;
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
        if (request()->ajax()) {
            if (request()->input('datatable') == 'payment') {
                return $this->_allPayments();
            }

            if (request()->input('datatable') == 'sell') {
                return $this->_allSales();
            }

            if (request()->input('datatable') == 'purchase') {
                return $this->_allPurchases();
            }

            if (request()->input('datatable') == 'expense') {
                return $this->_allExpenses();
            }


        }

        $business_id = request()->session()->get('user.business_id');

        $business_locations = BusinessLocation::forDropdown($business_id);
        $suppliers = Contact::suppliersDropdown($business_id, false);
        $orderStatuses = $this->transactionUtil->orderStatuses();

        return view('accounting::transactions.index')
            ->with(compact('business_locations', 'suppliers', 'orderStatuses'));
    }

    protected function _allSales()
    {
        $sale_type = 'sell';
        $business_id = request()->session()->get('user.business_id');

        $sells = $this->transactionUtil->getListSells($business_id, $sale_type);
        $sells->groupBy('transactions.id');

        $payment_types = $this->transactionUtil->payment_types(null, true, $business_id);
        $sales_order_statuses = Transaction::sales_order_statuses();

        $datatable = Datatables::of($sells)
                ->addColumn(
                    'action',
                    function ($row) {
                        $html = '';

                        if (auth()->user()->can('sell.view') || auth()->user()->can('direct_sell.view') || auth()->user()->can('view_own_sell_only')) {
                            $html .= '<a href="#" data-href="'.action([\App\Http\Controllers\SellController::class, 'show'], [$row->id]).'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-info" data-container=".view_modal"><i class="fas fa-eye" aria-hidden="true"></i> '.__('messages.view').'</a>';
                        }

                        if (auth()->user()->can('accounting.map_transactions')) {
                            //check if mapping already present
                            $is_mapped = AccountingAccountsTransaction::where('transaction_id', $row->id)->exists();

                            if (! $is_mapped) {
                                $html .= '<a href="#" 
                                    data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->id.'&type=sell'.'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-primary tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.map_transaction').'</a>';
                            } else {
                                $html .= '<a href="#" 
                                    data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->id.'&type=sell'.'" class="btn-modal btn btn-warning btn-xs tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.edit_mapping').'</a>';
                            }
                        }

                        return $html;
                    }
                )
                ->removeColumn('id')
                ->editColumn(
                    'final_total',
                    '<span class="final-total" data-orig-value="{{$final_total}}">@format_currency($final_total)</span>'
                )
                ->editColumn(
                    'tax_amount',
                    '<span class="total-tax" data-orig-value="{{$tax_amount}}">@format_currency($tax_amount)</span>'
                )
                ->editColumn(
                    'total_paid',
                    '<span class="total-paid" data-orig-value="{{$total_paid}}">@format_currency($total_paid)</span>'
                )
                ->editColumn(
                    'total_before_tax',
                    '<span class="total_before_tax" data-orig-value="{{$total_before_tax}}">@format_currency($total_before_tax)</span>'
                )
                ->editColumn(
                    'discount_amount',
                    function ($row) {
                        $discount = ! empty($row->discount_amount) ? $row->discount_amount : 0;

                        if (! empty($discount) && $row->discount_type == 'percentage') {
                            $discount = $row->total_before_tax * ($discount / 100);
                        }

                        return '<span class="total-discount" data-orig-value="'.$discount.'">'.$this->transactionUtil->num_f($discount, true).'</span>';
                    }
                )
                ->editColumn('transaction_date', '{{@format_datetime($transaction_date)}}')
                ->editColumn(
                    'payment_status',
                    function ($row) {
                        $payment_status = Transaction::getPaymentStatus($row);

                        return (string) view('sell.partials.payment_status', ['payment_status' => $payment_status, 'id' => $row->id]);
                    }
                )
                ->addColumn('total_remaining', function ($row) {
                    $total_remaining = $row->final_total - $row->total_paid;
                    $total_remaining_html = '<span class="payment_due" data-orig-value="'.$total_remaining.'">'.$this->transactionUtil->num_f($total_remaining, true).'</span>';

                    return $total_remaining_html;
                })
                ->addColumn('total_remaining', function ($row) {
                    $total_remaining = $row->final_total - $row->total_paid;
                    $total_remaining_html = '<span class="payment_due" data-orig-value="'.$total_remaining.'">'.$this->transactionUtil->num_f($total_remaining, true).'</span>';

                    return $total_remaining_html;
                })
                ->editColumn('invoice_no', function ($row) {
                    $invoice_no = $row->invoice_no;
                    if (! empty($row->woocommerce_order_id)) {
                        $invoice_no .= ' <i class="fab fa-wordpress text-primary no-print" title="'.__('lang_v1.synced_from_woocommerce').'"></i>';
                    }
                    if (! empty($row->return_exists)) {
                        $invoice_no .= ' &nbsp;<small class="label bg-red label-round no-print" title="'.__('lang_v1.some_qty_returned_from_sell').'"><i class="fas fa-undo"></i></small>';
                    }
                    if (! empty($row->is_recurring)) {
                        $invoice_no .= ' &nbsp;<small class="label bg-red label-round no-print" title="'.__('lang_v1.subscribed_invoice').'"><i class="fas fa-recycle"></i></small>';
                    }

                    if (! empty($row->recur_parent_id)) {
                        $invoice_no .= ' &nbsp;<small class="label bg-info label-round no-print" title="'.__('lang_v1.subscription_invoice').'"><i class="fas fa-recycle"></i></small>';
                    }

                    if (! empty($row->is_export)) {
                        $invoice_no .= '</br><small class="label label-default no-print" title="'.__('lang_v1.export').'">'.__('lang_v1.export').'</small>';
                    }

                    return $invoice_no;
                })

                ->addColumn('conatct_name', '@if(!empty($supplier_business_name)) {{$supplier_business_name}}, <br> @endif {{$name}}')
                ->editColumn('total_items', '{{@format_quantity($total_items)}}')
                ->filterColumn('conatct_name', function ($query, $keyword) {
                    $query->where(function ($q) use ($keyword) {
                        $q->where('contacts.name', 'like', "%{$keyword}%")
                        ->orWhere('contacts.supplier_business_name', 'like', "%{$keyword}%");
                    });
                })
                ->addColumn('payment_methods', function ($row) use ($payment_types) {
                    $methods = array_unique($row->payment_lines->pluck('method')->toArray());
                    $count = count($methods);
                    $payment_method = '';
                    if ($count == 1) {
                        $payment_method = $payment_types[$methods[0]] ?? '';
                    } elseif ($count > 1) {
                        $payment_method = __('lang_v1.checkout_multi_pay');
                    }

                    $html = ! empty($payment_method) ? '<span class="payment-method" data-orig-value="'.$payment_method.'" data-status-name="'.$payment_method.'">'.$payment_method.'</span>' : '';

                    return $html;
                })
                ->editColumn('status', function ($row) use ($sales_order_statuses) {
                    $status = '';

                    if ($row->type == 'sales_order') {
                        $status = '<span class="label '.$sales_order_statuses[$row->status]['class'].'" >'.$sales_order_statuses[$row->status]['label'].'</span>';
                    }

                    return $status;
                });

        $rawColumns = ['final_total', 'action', 'total_paid', 'total_remaining', 'payment_status', 'invoice_no', 'discount_amount', 'tax_amount', 'total_before_tax', 'shipping_status', 'types_of_service_name', 'payment_methods', 'return_due', 'conatct_name', 'status'];

        return $datatable->rawColumns($rawColumns)
                      ->make(true);
    }

    protected function _allPayments()
    {
        $transaction_type = request()->input('transaction_type');
        $business_id = request()->session()->get('user.business_id');

        $query = TransactionPayment::join(
                        'transactions as T',
                        'transaction_payments.transaction_id',
                        '=',
                        'T.id'
                    )
                    ->leftjoin('accounts as A', 'transaction_payments.account_id', '=', 'A.id')
                    ->where('transaction_payments.business_id', $business_id)
                    ->where('T.type', $transaction_type)
                    ->whereNull('transaction_payments.parent_id')
                    ->where('transaction_payments.method', '!=', 'advance')
                    ->leftjoin('contacts as c', 'transaction_payments.payment_for', '=', 'c.id')
                    ->select([
                        'paid_on',
                        'payment_ref_no',
                        'T.ref_no',
                        'T.invoice_no',
                        'T.type',
                        'T.id as transaction_id',
                        // 'A.name as account_name',
                        // 'A.account_number',
                        'transaction_payments.id as payment_id',
                        'transaction_payments.account_id',
                        'c.name as contact_name',
                        'c.type as contact_type',
                        'transaction_payments.is_advance',
                        'transaction_payments.amount',
                        'transaction_payments.id as transaction_payment_id',
                    ]);

        return DataTables::of($query)
                ->editColumn('paid_on', function ($row) {
                    return $this->transactionUtil->format_date($row->paid_on, true);
                })
                ->editColumn('amount', function ($row) {
                    return $this->transactionUtil->num_f($row->amount, true);
                })
                ->addColumn('details', function ($row) {
                    $details = '';

                    if ($row->contact_type == 'supplier') {
                        $details = '<b>'.__('role.supplier').':</b> '.$row->contact_name;
                    } else {
                        $details = '<b>'.__('role.customer').':</b> '.$row->contact_name;
                    }

                    return $details;
                })
                ->addColumn('action', function ($row) use ($transaction_type) {
                    $html = '';

                    if (auth()->user()->can('accounting.map_transactions')) {
                        //check if mapping already present
                        $is_mapped = AccountingAccountsTransaction::where('transaction_payment_id', $row->transaction_payment_id)->exists();

                        if ($transaction_type == 'purchase') {
                            $type_parameter = 'purchase_payment';
                        } elseif ($transaction_type == 'sell') {
                            $type_parameter = 'sell_payment';
                        }

                        if (! $is_mapped) {
                            $html .= '<a href="#" 
                                data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->transaction_payment_id.'&type='.$type_parameter.'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-primary tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.map_transaction').'</a>';
                        } else {
                            $html .= '<a href="#" 
                                data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->transaction_payment_id.'&type='.$type_parameter.'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-warning tw-dw-btn-xs mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.edit_mapping').'</a>';
                        }
                    }

                    return $html;
                })
                ->addColumn('transaction_number', function ($row) {
                    $html = $row->ref_no;
                    if ($row->type == 'sell') {
                        $html = '<button type="button" class="btn btn-link btn-modal"
                                data-href="'.action([\App\Http\Controllers\SellController::class, 'show'], [$row->transaction_id]).'" data-container=".view_modal">'.$row->invoice_no.'</button>';
                    } elseif ($row->type == 'purchase') {
                        $html = '<button type="button" class="btn btn-link btn-modal"
                                data-href="'.action([\App\Http\Controllers\PurchaseController::class, 'show'], [$row->transaction_id]).'" data-container=".view_modal">'.$row->ref_no.'</button>';
                    }

                    return $html;
                })
                ->editColumn('type', function ($row) {
                    $type = $row->type;
                    if ($row->type == 'sell') {
                        $type = __('sale.sale');
                    } elseif ($row->type == 'purchase') {
                        $type = __('lang_v1.purchase');
                    } elseif ($row->type == 'expense') {
                        $type = __('lang_v1.expense');
                    } elseif ($row->is_advance == 1) {
                        $type = __('lang_v1.advance');
                    }

                    return $type;
                })
                // ->filterColumn('account', function ($query, $keyword) {
                //     $query->where('A.name', 'like', ["%{$keyword}%"])
                //         ->orWhere('account_number', 'like', ["%{$keyword}%"]);
                // })
                ->filterColumn('transaction_number', function ($query, $keyword) {
                    $query->where('T.invoice_no', 'like', ["%{$keyword}%"])
                        ->orWhere('T.ref_no', 'like', ["%{$keyword}%"]);
                })
                ->rawColumns(['action', 'transaction_number', 'details'])
                ->make(true);
    }

    protected function _allPurchases()
    {
        $business_id = request()->session()->get('user.business_id');
        $purchases = $this->transactionUtil->getListPurchases($business_id);

        return Datatables::of($purchases)
                ->addColumn('action', function ($row) {
                    $html = '';

                    if (auth()->user()->can('purchase.view')) {
                        $html .= '<a href="#" data-href="'.action([\App\Http\Controllers\PurchaseController::class, 'show'], [$row->id]).'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-info tw-w-max" data-container=".view_modal"><i class="fas fa-eye" aria-hidden="true"></i> '.__('messages.view').'</a>';
                    }
                    if (auth()->user()->can('accounting.map_transactions')) {
                        //check if mapping already present
                        $is_mapped = AccountingAccountsTransaction::where('transaction_id', $row->id)->exists();

                        if (! $is_mapped) {
                            $html .= '<a href="#" 
                                data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->id.'&type=purchase'.'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-primary tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.map_transaction').'</a>';
                        } else {
                            $html .= '<a href="#" 
                                data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->id.'&type=purchase'.'" class="btn-modal btn btn-warning btn-xs tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.edit_mapping').'</a>';
                        }
                    }

                    return $html;
                })
                ->removeColumn('id')
                ->editColumn('ref_no', function ($row) {
                    return ! empty($row->return_exists) ? $row->ref_no.' <small class="label bg-red label-round no-print" title="'.__('lang_v1.some_qty_returned').'"><i class="fas fa-undo"></i></small>' : $row->ref_no;
                })
                ->editColumn(
                    'final_total',
                    '<span class="final_total" data-orig-value="{{$final_total}}">@format_currency($final_total)</span>'
                )
                ->editColumn('transaction_date', '{{@format_datetime($transaction_date)}}')
                ->editColumn('name', '@if(!empty($supplier_business_name)) {{$supplier_business_name}}, <br> @endif {{$name}}')
                ->editColumn(
                    'status',
                    '<span class="label @transaction_status($status) status-label" data-status-name="{{__(\'lang_v1.\' . $status)}}" data-orig-value="{{$status}}">{{__(\'lang_v1.\' . $status)}}
                        </span>'
                )
                ->editColumn(
                    'payment_status',
                    function ($row) {
                        $payment_status = Transaction::getPaymentStatus($row);

                        return (string) view('sell.partials.payment_status', ['payment_status' => $payment_status, 'id' => $row->id, 'for_purchase' => true]);
                    }
                )
                ->addColumn('payment_due', function ($row) {
                    $due = $row->final_total - $row->amount_paid;
                    $due_html = '<strong>'.__('lang_v1.purchase').':</strong> <span class="payment_due" data-orig-value="'.$due.'">'.$this->transactionUtil->num_f($due, true).'</span>';

                    if (! empty($row->return_exists)) {
                        $return_due = $row->amount_return - $row->return_paid;
                        $due_html .= '<br><strong>'.__('lang_v1.purchase_return').':</strong> <span class="purchase_return" data-orig-value="'.$return_due.'">'.$this->transactionUtil->num_f($return_due, true).'</span>';
                    }

                    return $due_html;
                })
                ->rawColumns(['final_total', 'action', 'payment_due', 'payment_status', 'status', 'ref_no', 'name'])
                ->make(true);
    }

    protected function _allExpenses()
    {
        $business_id = request()->session()->get('user.business_id');
        $expenses = $this->transactionUtil->getListExpenses($business_id);

        return Datatables::of($expenses)
        ->addColumn(
            'action',
            function ($row) {
                $html = '';
                if (auth()->user()->can('accounting.map_transactions')) {
                    //check if mapping already present
                    $is_mapped = AccountingAccountsTransaction::where('transaction_id', $row->id)->exists();

                    if (! $is_mapped) {
                        $html .= '<a href="#" 
                            data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->id.'&type=expense'.'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-primary tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.map_transaction').'</a>';
                    } else {
                        $html .= '<a href="#" 
                            data-href="'.action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'map']).'?id='.$row->id.'&type=expense'.'" class="btn-modal tw-dw-btn tw-dw-btn-xs tw-dw-btn-outline  tw-dw-btn-warning btn-xs tw-w-max mt-10" data-container=".view_modal"><i class="fas fa-link"></i> '.__('accounting::lang.edit_mapping').'</a>';
                    }
                }

                return $html;
            })
        ->removeColumn('id')
        ->editColumn(
            'final_total',
            '<span class="display_currency final-total" data-currency_symbol="true" data-orig-value="@if($type=="expense_refund"){{-1 * $final_total}}@else{{$final_total}}@endif">@if($type=="expense_refund") - @endif @format_currency($final_total)</span>'
        )
        ->editColumn('transaction_date', '{{@format_datetime($transaction_date)}}')
        ->editColumn(
            'payment_status',
            '<a href="{{ action([\App\Http\Controllers\TransactionPaymentController::class, \'show\'], [$id])}}" class="view_payment_modal payment-status" data-orig-value="{{$payment_status}}" data-status-name="{{__(\'lang_v1.\' . $payment_status)}}"><span class="label @payment_status($payment_status)">{{__(\'lang_v1.\' . $payment_status)}}
                </span></a>'
        )
        ->addColumn('payment_due', function ($row) {
            $due = $row->final_total - $row->amount_paid;

            if ($row->type == 'expense_refund') {
                $due = -1 * $due;
            }

            return '<span class="display_currency payment_due" data-currency_symbol="true" data-orig-value="'.$due.'">'.$this->transactionUtil->num_f($due, true).'</span>';
        })
        ->addColumn('recur_details', function ($row) {
            $details = '<small>';
            if ($row->is_recurring == 1) {
                $type = $row->recur_interval == 1 ? Str::singular(__('lang_v1.'.$row->recur_interval_type)) : __('lang_v1.'.$row->recur_interval_type);
                $recur_interval = $row->recur_interval.$type;

                $details .= __('lang_v1.recur_interval').': '.$recur_interval;
                if (! empty($row->recur_repetitions)) {
                    $details .= ', '.__('lang_v1.no_of_repetitions').': '.$row->recur_repetitions;
                }
                if ($row->recur_interval_type == 'months' && ! empty($row->subscription_repeat_on)) {
                    $details .= '<br><small class="text-muted">'.
                    __('lang_v1.repeat_on').': '.str_ordinal($row->subscription_repeat_on);
                }
            } elseif (! empty($row->recur_parent_id)) {
                $details .= __('lang_v1.recurred_from').': '.$row->recurring_parent->ref_no;
            }
            $details .= '</small>';

            return $details;
        })
        ->editColumn('ref_no', function ($row) {
            $ref_no = $row->ref_no;
            if (! empty($row->is_recurring)) {
                $ref_no .= ' &nbsp;<small class="label bg-red label-round no-print" title="'.__('lang_v1.recurring_expense').'"><i class="fas fa-recycle"></i></small>';
            }

            if (! empty($row->recur_parent_id)) {
                $ref_no .= ' &nbsp;<small class="label bg-info label-round no-print" title="'.__('lang_v1.generated_recurring_expense').'"><i class="fas fa-recycle"></i></small>';
            }

            if ($row->type == 'expense_refund') {
                $ref_no .= ' &nbsp;<small class="label bg-gray">'.__('lang_v1.refund').'</small>';
            }

            return $ref_no;
        })
        ->rawColumns(['final_total', 'action', 'payment_status', 'payment_due', 'ref_no', 'recur_details'])
        ->make(true);
    }

    public function map(Request $request)
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.map_transactions'))) {
            abort(403, 'Unauthorized action.');
        }

        if (request()->ajax()) {
            $type = $request->get('type');
            $id = $request->get('id');

            if ($type == 'sell') {
                $transaction = Transaction::where('id', $id)->where('business_id', $business_id)
                                    ->firstorFail();

                //setting defaults
                //if paid - Payment account = Sales
                //Deposit to = Account Receivable
                //Get all payment lines and map for each

                //if not paid - Payment account = Sales
                //Deposit to = Account Receivable

                $existing_payment = AccountingAccountsTransaction::where('transaction_id', $id)
                                        ->where('map_type', 'payment_account')
                                        ->first();
                $existing_deposit = AccountingAccountsTransaction::where('transaction_id', $id)
                                        ->where('map_type', 'deposit_to')
                                        ->first();
                $default_payment_account = ! empty($existing_payment) ? AccountingAccount::find($existing_payment->accounting_account_id) : null;
                $default_deposit_to = ! empty($existing_deposit) ? AccountingAccount::find($existing_deposit->accounting_account_id) : null;
                $note = ! empty($existing_deposit) ? $existing_deposit->note  : null;

                return view('accounting::transactions.map')
                        ->with(compact('transaction', 'type', 'default_payment_account', 'default_deposit_to', 'note'));
            } elseif (in_array($type, ['purchase_payment', 'sell_payment'])) {
                $transaction_payment = TransactionPayment::where('id', $id)->where('business_id', $business_id)
                                    ->firstorFail();

                $existing_payment = AccountingAccountsTransaction::where('transaction_payment_id', $id)
                                    ->where('map_type', 'payment_account')
                                    ->first();
                $existing_deposit = AccountingAccountsTransaction::where('transaction_payment_id', $id)
                                    ->where('map_type', 'deposit_to')
                                    ->first();
                $default_payment_account = ! empty($existing_payment) ? AccountingAccount::find($existing_payment->accounting_account_id) : null;
                $default_deposit_to = ! empty($existing_deposit) ? AccountingAccount::find($existing_deposit->accounting_account_id) : null;
                $note = ! empty($existing_deposit) ? $existing_deposit->note  : null;

                return view('accounting::transactions.map')
                            ->with(compact('transaction_payment', 'type', 'default_payment_account', 'default_deposit_to', 'note'));
            } elseif ($type == 'purchase') {
                $transaction = Transaction::where('id', $id)->where('business_id', $business_id)
                                    ->firstorFail();

                //setting defaults
                //if paid - Payment account = Sales
                //Deposit to = Account Receivable
                //Get all payment lines and map for each

                //if not paid - Payment account = Sales
                //Deposit to = Account Receivable

                $existing_payment = AccountingAccountsTransaction::where('transaction_id', $id)
                                        ->where('map_type', 'payment_account')
                                        ->first();
                $existing_deposit = AccountingAccountsTransaction::where('transaction_id', $id)
                                        ->where('map_type', 'deposit_to')
                                        ->first();
                $default_payment_account = ! empty($existing_payment) ? AccountingAccount::find($existing_payment->accounting_account_id) : null;
                $default_deposit_to = ! empty($existing_deposit) ? AccountingAccount::find($existing_deposit->accounting_account_id) : null;
                $note = ! empty($existing_deposit) ? $existing_deposit->note  : null;

                return view('accounting::transactions.map')
                        ->with(compact('transaction', 'type', 'default_payment_account', 'default_deposit_to', 'note'));
            } elseif ($type == 'expense') {
                $transaction = Transaction::where('id', $id)->where('business_id', $business_id)
                                    ->firstorFail();

                //setting defaults
                //if paid - Payment account = Sales
                //Deposit to = Account Receivable
                //Get all payment lines and map for each

                //if not paid - Payment account = Sales
                //Deposit to = Account Receivable

                $existing_payment = AccountingAccountsTransaction::where('transaction_id', $id)
                                        ->where('map_type', 'payment_account')
                                        ->first();
                $existing_deposit = AccountingAccountsTransaction::where('transaction_id', $id)
                                        ->where('map_type', 'deposit_to')
                                        ->first();
                $default_payment_account = ! empty($existing_payment) ? AccountingAccount::find($existing_payment->accounting_account_id) : null;
                $default_deposit_to = ! empty($existing_deposit) ? AccountingAccount::find($existing_deposit->accounting_account_id) : null;

                $note = ! empty($existing_deposit) ? $existing_deposit->note  : null;

                return view('accounting::transactions.map')
                        ->with(compact('transaction', 'type', 'default_payment_account', 'default_deposit_to', 'note'));
            }

        }
    }

    public function saveMap(Request $request)
    {
        $business_id = request()->session()->get('user.business_id');

        if (! (auth()->user()->can('superadmin') ||
            $this->moduleUtil->hasThePermissionInSubscription($business_id, 'accounting_module')) ||
            ! (auth()->user()->can('accounting.map_transactions'))) {
            abort(403, 'Unauthorized action.');
        }

        try {
            if (request()->ajax()) {
                DB::beginTransaction();

                $type = $request->get('type');
                $id = $request->get('id');
                $user_id = request()->session()->get('user.id');

                $deposit_to = $request->get('deposit_to');
                $payment_account = $request->get('payment_account');
                $note= $request->get('description');

                $this->accountingUtil->saveMap($type, $id, $user_id, $business_id, $deposit_to, $payment_account, $note);

                DB::commit();

                $output = ['success' => true,
                    'msg' => __('lang_v1.updated_success'),
                ];
            }
        } catch (\Exception $e) {
            print_r($e->getMessage());
            exit;
            DB::rollBack();

            \Log::emergency('File:'.$e->getFile().'Line:'.$e->getLine().'Message:'.$e->getMessage());

            $output = ['success' => false,
                'msg' => __('messages.something_went_wrong'),
            ];
        }

        return $output;
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return Response
     */
    public function create()
    {
        return view('accounting::create');
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  Request  $request
     * @return Response
     */
    public function store(Request $request)
    {
        //
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
