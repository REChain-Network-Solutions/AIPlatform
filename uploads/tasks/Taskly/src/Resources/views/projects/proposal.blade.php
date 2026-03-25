@extends('layouts.main')
@section('page-title')
    {{ __('Manage Proposal') }}
@endsection
@section('page-breadcrumb')
    {{ __('Proposal') }}
@endsection
@push('scripts')
    <script>
        $(document).ready(function() {
            $('.cp_link').on('click', function() {
                var value = $(this).attr('data-link');
                var $temp = $("<input>");
                $("body").append($temp);
                $temp.val(value).select();
                document.execCommand("copy");
                $temp.remove();
                toastrs('Success', '{{ __('Link Copy on Clipboard') }}', 'success')
            });
        });
    </script>
@endpush
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
                                {{ __('Proposal') }}
                            </h5>
                        </div>
                        @permission('proposal create')
                            <div class=" col-1 text-end">
                                <a href="{{ route('proposal.create', ['cid' => 0, 'type' => 'project', 'project_id' => $project->id, 'redirect_route' => route('projects.proposal', $project->id)]) }}
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
                                    <th> {{ __('Proposal') }}</th>
                                    @if (\Auth::user()->type != 'client')
                                        <th> {{ __('Customer') }}</th>
                                    @endif
                                    <th>{{ __('Account Type') }}</th>
                                    <th> {{ __('Issue Date') }}</th>
                                    <th> {{ __('Status') }}</th>
                                    <th> {{ __('Action') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach ($proposals as $proposal)
                                    <tr class="font-style">
                                        <td class="Id">
                                            <a href="{{ route('proposal.show', \Crypt::encrypt($proposal->id)) }}"
                                                class="btn btn-outline-primary">{{ \App\Models\Proposal::proposalNumberFormat($proposal->proposal_id) }}
                                            </a>
                                        </td>
                                        @if (\Auth::user()->type != 'client')
                                            <td> {{ !empty($proposal->customer) ? $proposal->customer->name : '' }} </td>
                                        @endif
                                        <td>{{ $proposal->account_type }}</td>
                                        <td>{{ company_date_formate($proposal->issue_date) }}</td>
                                        <td>
                                            @if ($proposal->status == 0)
                                                <span
                                                    class="badge fix_badge bg-primary p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                            @elseif($proposal->status == 1)
                                                <span
                                                    class="badge fix_badge bg-info p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                            @elseif($proposal->status == 2)
                                                <span
                                                    class="badge fix_badge bg-secondary p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                            @elseif($proposal->status == 3)
                                                <span
                                                    class="badge fix_badge bg-warning p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                            @elseif($proposal->status == 4)
                                                <span
                                                    class="badge fix_badge bg-danger p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                            @endif
                                        </td>
                                        <td class="Action">
                                            <span>
                                                <div class="action-btn me-2">
                                                    <a href="#" class="btn btn-sm  align-items-center cp_link bg-primary"
                                                        data-link="{{ route('pay.proposalpay', \Illuminate\Support\Facades\Crypt::encrypt($proposal->id)) }}"
                                                        data-bs-toggle="tooltip" title="{{ __('Copy') }}"
                                                        data-original-title="{{ __('Click to copy proposal link') }}">
                                                        <i class="ti ti-file text-white"></i>
                                                    </a>
                                                </div>
                                                @if ($proposal->is_convert == 0 && $proposal->is_convert_retainer == 0)
                                                    @permission('proposal convert invoice')
                                                        <div class="action-btn me-2">
                                                            {!! Form::open([
                                                                'method' => 'get',
                                                                'route' => ['proposal.convert', $proposal->id],
                                                                'id' => 'proposal-form-' . $proposal->id,
                                                            ]) !!}
                                                            <a href="#"
                                                                class="mx-3 btn btn-sm  align-items-center bs-pass-para show_confirm bg-success"
                                                                data-bs-toggle="tooltip" title=""
                                                                data-bs-original-title="{{ __('Convert to Invoice') }}"
                                                                aria-label="Delete"
                                                                data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                data-confirm-yes="proposal-form-{{ $proposal->id }}">
                                                                <i class="ti ti-exchange text-white"></i>
                                                            </a>
                                                            {{ Form::close() }}
                                                        </div>
                                                    @endpermission
                                                @elseif($proposal->is_convert == 1)
                                                    @permission('invoice show')
                                                        <div class="action-btn me-2">
                                                            <a href="{{ route('invoice.show', \Crypt::encrypt($proposal->converted_invoice_id)) }}"
                                                                class="mx-3 btn btn-sm  align-items-center bg-success"
                                                                data-bs-toggle="tooltip"
                                                                title="{{ __('Already convert to Invoice') }}">
                                                                <i class="ti ti-eye text-white"></i>
                                                            </a>
                                                        </div>
                                                    @endpermission
                                                @endif
                                                @if (module_is_active('Retainer'))
                                                    @include('retainer::setting.convert_retainer', [
                                                        'proposal' => $proposal,
                                                        'type' => 'list',
                                                    ])
                                                @endif
                                                @permission('proposal duplicate')
                                                    <div class="action-btn me-2">
                                                        {!! Form::open([
                                                            'method' => 'get',
                                                            'route' => ['proposal.duplicate', $proposal->id],
                                                            'id' => 'duplicate-form-' . $proposal->id,
                                                        ]) !!}
                                                        <a href="#"
                                                            class="mx-3 btn btn-sm  align-items-center bs-pass-para show_confirm bg-secondary"
                                                            data-bs-toggle="tooltip" title=""
                                                            data-bs-original-title="{{ __('Duplicate') }}" aria-label="Delete"
                                                            data-text="{{ __('You want to confirm duplicate this proposal. Press Yes to continue or Cancel to go back') }}"
                                                            data-confirm-yes="duplicate-form-{{ $proposal->id }}">
                                                            <i class="ti ti-copy text-white text-white"></i>
                                                        </a>
                                                        {{ Form::close() }}
                                                    </div>
                                                @endpermission
                                                @permission('proposal show')
                                                    <div class="action-btn me-2">
                                                        <a href="{{ route('proposal.show', \Crypt::encrypt($proposal->id)) }}"
                                                            class="mx-3 btn btn-sm align-items-center bg-warning" data-bs-toggle="tooltip"
                                                            title="{{ __('Show') }}"
                                                            data-original-title="{{ __('Detail') }}">
                                                            <i class="ti ti-eye text-white text-white"></i>
                                                        </a>
                                                    </div>
                                                @endpermission
                                                @if (module_is_active('ProductService') &&
                                                        ($proposal->proposal_module == 'taskly' ? module_is_active('Taskly') : module_is_active('Account')) &&
                                                        ($proposal->proposal_module == 'cmms' ? module_is_active('CMMS') : module_is_active('Account')))
                                                    @permission('proposal edit')
                                                        <div class="action-btn me-2">
                                                            <a href="{{ route('proposal.edit', \Crypt::encrypt($proposal->id)) }}"
                                                                class="mx-3 btn btn-sm  align-items-center bg-info"
                                                                data-bs-toggle="tooltip"
                                                                data-bs-original-title="{{ __('Edit') }}">
                                                                <i class="ti ti-pencil text-white"></i>
                                                            </a>
                                                        </div>
                                                    @endpermission
                                                @endif
                                                @permission('proposal delete')
                                                    <div class="action-btn">
                                                        {{ Form::open(['route' => ['proposal.destroy', $proposal->id], 'class' => 'm-0']) }}
                                                        @method('DELETE')
                                                        <a href="#"
                                                            class="mx-3 btn btn-sm  align-items-center bs-pass-para show_confirm bg-danger"
                                                            data-bs-toggle="tooltip" title=""
                                                            data-bs-original-title="Delete" aria-label="Delete"
                                                            data-confirm="{{ __('Are You Sure?') }}"
                                                            data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                            data-confirm-yes="delete-form-{{ $proposal->id }}"><i
                                                                class="ti ti-trash text-white text-white"></i></a>
                                                        {{ Form::close() }}
                                                    </div>
                                                @endpermission
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
