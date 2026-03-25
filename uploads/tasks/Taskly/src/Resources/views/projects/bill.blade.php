@extends('layouts.main')
@section('page-title')
    {{ __('Manage Bug Stages') }}
@endsection
@section('page-action')
@endsection
@section('content')
    <div id="retainer-settings" class="row">
        <div class="col-md-3">
            @include('taskly::layouts.finance_tab')
        </div>
        <div class="col-md-9">
            <div class="card retainer">
                <div class="card-header">
                    <div class="row">
                        <div class="col-11">
                            <h5 class="">
                                {{ __('Invoice') }}
                            </h5>
                        </div>
                        @permission('bill create')
                            <div class=" col-1 text-end">
                                <a href="{{ route('bills.create', ['cid' => 0, 'type' => 'project', 'project_id' => $project->id]) }}
                                    "
                                    class="btn btn-sm btn-primary" data-bs-toggle="tooltip"
                                    data-bs-original-title="{{ __('Create') }}">
                                    <i class="ti ti-plus"></i>
                                </a>
                            </div>
                        @endpermission
                    </div>
                </div>
                <div class="card-body table-border-style">
                    <div class="table-responsive">
                        <table class="table mb-0 pc-dt-simple" id="assets">
                            <thead>
                                <tr>
                                    <th> {{ __('Invoice') }}</th>
                                    <th>{{ __('Account Type') }}</th>
                                    <th>{{ __('Issue Date') }}</th>
                                    <th>{{ __('Due Date') }}</th>
                                    <th>{{ __('Total Amount') }}</th>
                                    <th>{{ __('Due Amount') }}</th>
                                    <th>{{ __('Status') }}</th>
                                    <th>{{ __('Action') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach ($invoices as $invoice)
                                    <tr>
                                        <td class="Id">
                                            @if (Laratrust::hasPermission('invoice show'))
                                                <a href="{{ route('invoice.show', \Crypt::encrypt($invoice->id)) }}"
                                                    class="btn btn-outline-primary">{{ App\Models\Invoice::invoiceNumberFormat($invoice->invoice_id) }}</a>
                                            @else
                                                <a href="#"
                                                    class="btn btn-outline-primary">{{ App\Models\Invoice::invoiceNumberFormat($invoice->invoice_id) }}</a>
                                            @endif
                                        </td>

                                        <td>{{ $invoice->account_type == 'Taskly' ? 'Project' : preg_replace('/([a-z])([A-Z])/', '$1 $2', $invoice->account_type) }}
                                        </td>
                                        <td>{{ company_date_formate($invoice->issue_date) }}</td>
                                        <td>
                                            @if ($invoice->due_date < date('Y-m-d'))
                                                <p class="text-danger">
                                                    {{ company_date_formate($invoice->due_date) }}</p>
                                            @else
                                                {{ company_date_formate($invoice->due_date) }}
                                            @endif
                                        </td>

                                        <td>{{ currency_format_with_sym($invoice->getTotal()) }}</td>


                                        <td>{{ currency_format_with_sym($invoice->getDue()) }}</td>


                                        <td>
                                            @if ($invoice->status == 0)
                                                <span
                                                    class="badge fix_badges bg-primary p-2 px-3 rounded">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                            @elseif($invoice->status == 1)
                                                <span
                                                    class="badge fix_badges bg-info p-2 px-3 rounded">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                            @elseif($invoice->status == 2)
                                                <span
                                                    class="badge fix_badges bg-secondary p-2 px-3 rounded">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                            @elseif($invoice->status == 3)
                                                <span
                                                    class="badge fix_badges bg-warning p-2 px-3 rounded">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                            @elseif($invoice->status == 4)
                                                <span
                                                    class="badge fix_badges bg-danger p-2 px-3 rounded">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                            @endif
                                        </td>

                                        <td class="Action">
                                            <span>
                                                <div class="action-btn bg-primary ms-2">
                                                    <a href="#" class="btn btn-sm  align-items-center cp_link"
                                                        data-link="{{ route('pay.invoice', \Illuminate\Support\Facades\Crypt::encrypt($invoice->id)) }}"
                                                        data-bs-toggle="tooltip" title="{{ __('Copy') }}"
                                                        data-original-title="{{ __('Click to copy invoice link') }}">
                                                        <i class="ti ti-file text-white"></i>
                                                    </a>
                                                </div>
                                                @if (module_is_active('EInvoice'))
                                                    @permission('download invoice')
                                                        @include('einvoice::download.generate_invoice', [
                                                            'invoice_id' => $invoice->id,
                                                        ])
                                                    @endpermission
                                                @endif
                                                <div class="action-btn bg-info ms-2">
                                                    <a href="#" class="btn btn-sm  align-items-center"
                                                        data-url="{{ route('delivery-form.pdf', \Crypt::encrypt($invoice->id)) }}"
                                                        data-ajax-popup="true" data-size="lg" data-bs-toggle="tooltip"
                                                        title="{{ __('Invoice Delivery Form') }}"
                                                        data-title="{{ __('Invoice Delivery Form') }}">
                                                        <i class="ti ti-clipboard-list text-white"></i>
                                                    </a>
                                                </div>

                                                @permission('invoice duplicate')
                                                    <div class="action-btn bg-secondary ms-2">
                                                        {!! Form::open([
                                                            'method' => 'get',
                                                            'route' => ['invoice.duplicate', $invoice->id],
                                                            'id' => 'duplicate-form-' . $invoice->id,
                                                        ]) !!}
                                                        <a href="#"
                                                            class="mx-3 btn btn-sm  align-items-center bs-pass-para show_confirm"
                                                            data-bs-toggle="tooltip" title=""
                                                            data-bs-original-title="{{ __('Duplicate') }}" aria-label="Delete"
                                                            data-text="{{ __('You want to confirm duplicate this invoice. Press Yes to continue or Cancel to go back') }}"
                                                            data-confirm-yes="duplicate-form-{{ $invoice->id }}">
                                                            <i class="ti ti-copy  text-white"></i>
                                                        </a>
                                                        {{ Form::close() }}
                                                    </div>
                                                @endpermission
                                                @permission('invoice show')
                                                    <div class="action-btn bg-warning ms-2">
                                                        <a href="{{ route('invoice.show', \Crypt::encrypt($invoice->id)) }}"
                                                            class="mx-3 btn btn-sm align-items-center" data-bs-toggle="tooltip"
                                                            title="{{ __('View') }}">
                                                            <i class="ti ti-eye  text-white"></i>
                                                        </a>
                                                    </div>
                                                @endpermission


                                                @if ($invoice->status != 4)
                                                    @permission('invoice edit')
                                                        <div class="action-btn bg-info ms-2">
                                                            <a href="{{ route('invoice.edit', \Crypt::encrypt($invoice->id)) }}"
                                                                class="mx-3 btn btn-sm  align-items-center"
                                                                data-bs-toggle="tooltip"
                                                                data-bs-original-title="{{ __('Edit') }}">
                                                                <i class="ti ti-pencil text-white"></i>
                                                            </a>
                                                        </div>
                                                    @endpermission

                                                    @permission('invoice delete')
                                                        <div class="action-btn bg-danger ms-2">
                                                            {{ Form::open(['route' => ['invoice.destroy', $invoice->id], 'class' => 'm-0']) }}
                                                            @method('DELETE')
                                                            <a href="#"
                                                                class="mx-3 btn btn-sm  align-items-center bs-pass-para show_confirm"
                                                                data-bs-toggle="tooltip" title=""
                                                                data-bs-original-title="Delete" aria-label="Delete"
                                                                data-confirm="{{ __('Are You Sure?') }}"
                                                                data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                data-confirm-yes="delete-form-{{ $invoice->id }}">
                                                                <i class="ti ti-trash text-white text-white"></i>
                                                            </a>
                                                            {{ Form::close() }}
                                                        </div>
                                                    @endpermission
                                                @endif

                                            </span>
                                        </td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection
