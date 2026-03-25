@extends('layouts.app')
@section('page-title')
    {{ __('Workorder') }}
@endsection
@section('breadcrumb')
    <ul class="breadcrumb mb-0">
        <li class="breadcrumb-item">
            <a href="{{ route('dashboard') }}">{{ __('Dashboard') }}</a>
        </li>
        <li class="breadcrumb-item">
            <a href="{{ route('workorder.index') }}">{{ __('Workorder') }}</a>
        </li>
        <li class="breadcrumb-item active">
            <a href="#">{{ workOrderPrefix() . $workorder->wo_id }}</a>
        </li>
    </ul>
@endsection
@php
    $admin_logo = getSettingsValByName('company_logo');
    $settings = settings();
@endphp
@push('script-page')
    <script>
        $(document).on('click', '.print', function() {
            var printContents = document.getElementById('invoice-print').innerHTML;
            var originalContents = document.body.innerHTML;
            document.body.innerHTML = printContents;
            $('.invoice-action').addClass('d-none');
            window.print();
            $('.invoice-action').removeClass('d-none');
            document.body.innerHTML = originalContents;
        });

        $(document).on('click', '.workorderStatusChange', function() {
            var workorderStatus = this.value;
            var workorderUrl = $(this).data('url');
            $.ajax({
                url: workorderUrl + '?status=' + workorderStatus,
                type: 'GET',
                cache: false,
                success: function(data) {
                    location.reload();
                },
            });
        });
    </script>
@endpush


@section('content')
    <div class="row">
        <!-- [ sample-page ] start -->
        <div class="col-sm-12">
            <div class="card">
                <div class="card-body">
                    <ul class="nav nav-tabs profile-tabs border-bottom mb-3 d-print-none" id="myTab" role="tablist">
                        <li class="nav-item">
                            <a class="nav-link active" id="profile-tab-1" data-bs-toggle="tab" href="#profile-1"
                                role="tab" aria-selected="true">
                                {{ __('Services and Parts') }}
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" id="profile-tab-2" data-bs-toggle="tab" href="#profile-2" role="tab"
                                aria-selected="true">
                                {{ __('Service Tasks') }}
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" id="profile-tab-3" data-bs-toggle="tab" href="#profile-3" role="tab"
                                aria-selected="true">
                                {{ __('Service Appointment') }}
                            </a>
                        </li>
                    </ul>
                    <div class="tab-content">
                        <div class="tab-pane show active" id="profile-1" role="tabpanel" aria-labelledby="profile-tab-1">
                            <div class="card border">
                                <div class="card-header">
                                    <div class="row">
                                        <div class="col-sm-6">
                                            <h5 class="mb-0">{{ __('Services and Parts') }}</h5>
                                        </div>
                                        <div class="col-sm-6 text-sm-end">
                                            <a href="javascript:void(0);" class="avtar avtar-s btn-link-secondary print"
                                                data-bs-toggle="tooltip" data-bs-original-title="{{ __('Download') }}">
                                                <i class="ph-duotone ph-printer f-22"></i>
                                            </a>
                                        </div>
                                    </div>
                                </div>
                                <div class="row" id="invoice-print">


                                    <div class="card-body">
                                        <div class="row g-3">
                                            <div class="col-12 ">
                                                <div class="row align-items-center g-3">
                                                    <div class="col-sm-6 ">
                                                        <div
                                                            class="d-flex align-items-center mb-2 navbar-brand img-fluid invoice-logo">
                                                            <img src="{{ asset(Storage::url('upload/logo/')) . '/' . (isset($admin_logo) && !empty($admin_logo) ? $admin_logo : 'logo.png') }}"
                                                                class="img-fluid brand-logo" alt="images" />
                                                        </div>
                                                        <p class="mb-0">{{ workOrderPrefix() . $workorder->wo_id }}</p>
                                                    </div>
                                                    <div class="col-sm-6 text-sm-end">

                                                        <h6>
                                                            {{ __('Assign To') }} :
                                                            <span
                                                                class="text-muted f-w-400">{{ !empty($workorder->assigned) ? $workorder->assigned->name : '-' }}
                                                            </span>
                                                        </h6>
                                                        <h6>
                                                            {{ __('Asset') }} :
                                                            <span
                                                                class="text-muted f-w-400">{{ !empty($workorder->assets) ? $workorder->assets->name : '-' }}</span>
                                                        </h6>
                                                        <h6>
                                                            {{ __('Type') }} :
                                                            <span
                                                                class="text-muted f-w-400">{{ !empty($workorder->types) ? $workorder->types->type : '-' }}</span>
                                                        </h6>
                                                        <h6>
                                                            {{ __('Due Date') }} :
                                                            <span
                                                                class="text-muted f-w-400">{{ dateFormat($workorder->due_date) }}</span>
                                                        </h6>
                                                        <h6>
                                                            {{ __('Status') }} :
                                                            <span class="text-muted f-w-400">
                                                                @if ($workorder->status == 'pending')
                                                                    <span
                                                                        class="badge text-bg-warning">{{ \App\Models\Estimation::$status[$workorder->status] }}</span>
                                                                @elseif($workorder->status == 'on_hold')
                                                                    <span
                                                                        class="badge text-bg-primary">{{ \App\Models\Estimation::$status[$workorder->status] }}</span>
                                                                @elseif($workorder->status == 'approved' || $workorder->status == 'completed')
                                                                    <span
                                                                        class="badge text-bg-success">{{ \App\Models\Estimation::$status[$workorder->status] }}</span>
                                                                @else
                                                                    <span
                                                                        class="badge text-bg-danger">{{ \App\Models\Estimation::$status[$workorder->status] }}</span>
                                                                @endif
                                                            </span>
                                                        </h6>
                                                        <h6>
                                                            {{ __('Priority') }} :
                                                            <span class="text-muted f-w-400">
                                                                @if ($workorder->priority == 'low')
                                                                    <span
                                                                        class="badge text-bg-primary">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                                                @elseif($workorder->priority == 'medium')
                                                                    <span
                                                                        class="badge text-bg-info">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                                                @elseif($workorder->priority == 'high')
                                                                    <span
                                                                        class="badge text-bg-warning">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                                                @elseif($workorder->priority == 'critical')
                                                                    <span
                                                                        class="badge text-bg-danger">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                                                @endif
                                                            </span>
                                                        </h6>

                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-sm-6 col-md-4">
                                                <div class="border rounded p-3">
                                                    <h6 class="mb-0">{{ __('From') }} :</h6>
                                                    <h5>{{ $settings['company_name'] }}</h5>
                                                    <p class="mb-0">{{ $settings['company_phone'] }}</p>
                                                    <p class="mb-0">{{ $settings['company_email'] }}</p>
                                                </div>
                                            </div>
                                            <div
                                                class="col-sm-4 col-md {{ !empty($workorder->clients->clients->billing_address) ? '4' : '6' }}">
                                                <div class="border rounded p-3">
                                                    <h6 class="mb-0">{{ __('To') }} :</h6>
                                                    <h5>{{ !empty($invoice) && !empty($invoice->Customer) ? $invoice->Customer->name : '' }}
                                                    </h5>
                                                    <p class="mb-0">
                                                        {{ !empty($workorder->clients) ? $workorder->clients->name : '' }}
                                                        ({{ !empty($workorder->clients) && !empty($workorder->clients->clients) ? $workorder->clients->clients->company : '' }})
                                                    </p>

                                                    <p class="mb-0">
                                                        {{ !empty($workorder->clients) ? $workorder->clients->phone_number : '' }}
                                                    </p>

                                                    <p class="mb-0">
                                                    <h6 class="mt-10 text-primary">{{ __('Service Address') }}:</h6>

                                                    {{ !empty($workorder->clients) && !empty($workorder->clients->clients) ? $workorder->clients->clients->service_address : '' }}
                                                    @if (
                                                        !empty($workorder->clients) &&
                                                            !empty($workorder->clients->clients) &&
                                                            !empty($workorder->clients->clients->service_city))
                                                        <br> {{ $workorder->clients->clients->service_city }}
                                                        , {{ $workorder->clients->clients->service_state }}
                                                        , {{ $workorder->clients->clients->service_country }},
                                                        {{ $workorder->clients->clients->service_zip_code }}
                                                    @endif
                                                    </p>
                                                </div>
                                            </div>

                                            @if (!empty($workorder->clients->clients->billing_address))
                                                <div class="col-sm-12 col-md-4">
                                                    <div class="border rounded p-3">
                                                        <h6 class="mt-10 text-primary">{{ __('Billing Address') }}:</h6>

                                                        {{ !empty($workorder->clients) && !empty($workorder->clients->clients) ? $workorder->clients->clients->billing_address : '' }}
                                                        @if (
                                                            !empty($workorder->clients) &&
                                                                !empty($workorder->clients->clients) &&
                                                                !empty($workorder->clients->clients->billing_city))
                                                            <br> {{ $workorder->clients->clients->billing_city }}
                                                            , {{ $workorder->clients->clients->billing_state }}
                                                            , {{ $workorder->clients->clients->billing_country }},
                                                            {{ $workorder->clients->clients->billing_zip_code }}
                                                        @endif
                                                    </div>
                                                </div>
                                            @endif

                                            <div class="col-12">
                                                <div class="table-responsive">
                                                    <table class="table table-hover mb-0">
                                                        <thead>
                                                            <tr>
                                                                <th>{{ __('Service') }}</th>
                                                                <th>{{ __('Quantity') }}</th>
                                                                <th>{{ __('Description') }}</th>
                                                                <th>{{ __('Amount') }}</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>

                                                            @foreach ($workorder->services as $service)
                                                                <tr>
                                                                    <td>{{ !empty($service->serviceParts) ? $service->serviceParts->title : '-' }}
                                                                    </td>
                                                                    <td>{{ $service->quantity }}
                                                                        {{ !empty($service->serviceParts) ? $service->serviceParts->unit : '' }}
                                                                    </td>
                                                                    <td>{{ !empty($service->description) ? $service->description : '-' }}
                                                                    </td>
                                                                    <td>{{ priceFormat($service->amount) }}</td>
                                                                </tr>
                                                            @endforeach
                                                        </tbody>
                                                    </table>
                                                </div>
                                                <div class="text-start">
                                                    <hr class="mb-2 mt-1 border-secondary border-opacity-50" />
                                                </div>
                                            </div>



                                            <div class="col-12 mt-5">
                                                <div class="table-responsive">
                                                    <table class="table table-hover mb-0">
                                                        <thead>
                                                            <tr>
                                                                <th>{{ __('Part') }}</th>
                                                                <th>{{ __('Quantity') }}</th>
                                                                <th>{{ __('Description') }}</th>
                                                                <th>{{ __('Amount') }}</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            @foreach ($workorder->parts as $part)
                                                                <tr>
                                                                    <td>
                                                                        {{ !empty($part->serviceParts) ? $part->serviceParts->title : '-' }}
                                                                    </td>
                                                                    <td>{{ $part->quantity }}
                                                                        {{ !empty($part->serviceParts) ? $part->serviceParts->unit : '' }}
                                                                    </td>
                                                                    <td>{{ !empty($part->description) ? $part->description : '-' }}
                                                                    </td>
                                                                    <td>{{ priceFormat($part->amount) }}</td>

                                                                </tr>
                                                            @endforeach
                                                        </tbody>
                                                    </table>
                                                </div>
                                                <div class="text-start">
                                                    <hr class="mb-2 mt-1 border-secondary border-opacity-50" />
                                                </div>
                                            </div>

                                            <div class="card-body p-3">
                                                <div class="rounded p-3 bg-light-secondary">
                                                    <div class="row justify-content-end">
                                                        <div class="col-auto">
                                                            <div class="table-responsive">
                                                                <table class="table table-borderless text-end mb-0">
                                                                    <tbody>

                                                                        <tr>
                                                                            <td class="pe-0 pt-0">

                                                                                <h4 class="text-primary m-r-10 ">
                                                                                    {{ __('Grand Total') }} :</h4>
                                                                            </td>
                                                                            <td class="ps-0 pt-0">

                                                                                <h4 class="text-primary">
                                                                                    {{ priceFormat($workorder->getWorkorderTotalAmount()) }}
                                                                                </h4>
                                                                            </td>
                                                                        </tr>

                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div class="col-12 text-dark">
                                                        @if (!empty($workorder->notes))
                                                            {{ __('Notes') }} : <p>{{ $workorder->notes }}</p>
                                                        @endif
                                                    </div>
                                                    @if (\Auth::user()->type == 'owner')
                                                        <div class="row mb-2">
                                                            @foreach ($status as $k => $val)
                                                                <div class="col-md-3 col-xxl-2">
                                                                    <div class="card border p-3">
                                                                        <div class="form-check">
                                                                            <input
                                                                                class="form-check-input estimationStatusChange"
                                                                                type="radio"
                                                                                value="{{ $k }}"
                                                                                {{ $workorder->status == $k ? 'checked' : '' }}
                                                                                id="{{ $val }}"
                                                                                data-url="{{ route('estimation.status', $workorder->id) }}"
                                                                                name="status"></span>


                                                                            <label class="form-check-label d-block"
                                                                                for="{{ $val }}">
                                                                                <span
                                                                                    class="h5 mb-0 d-block mt-1">{{ $val }}</span>

                                                                            </label>
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            @endforeach
                                                        </div>
                                                    @endif
                                                </div>
                                            </div>
                                        </div>
                                    </div>


                                </div>
                            </div>
                        </div>



                        <div class="tab-pane" id="profile-2" role="tabpanel" aria-labelledby="profile-tab-2">
                            <div class="card border">
                                <div class="card-header">
                                    <div class="row align-items-center">
                                        <div class="col-sm-6">
                                            <h5 class="mb-0">{{ __('Service Task List') }}</h5>
                                        </div>
                                        <div class="col-sm-6 text-sm-end">
                                            @if (Gate::check('create workorder service task'))
                                                <a class="btn btn-secondary btn-sm customModal mt-1" href="#"
                                                    data-url="{{ route('workorder.service.task.create', $workorder->id) }}"
                                                    data-size="md" data-title="{{ __('Create Service Task') }}">
                                                    <i class="ti ti-circle-plus align-text-bottom"></i>
                                                    {{ __('Create Task') }}
                                                </a>
                                            @endif
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body pt-0">
                                    <div class="dt-responsive table-responsive">
                                        <table class="table table-hover">
                                            <thead>
                                                <tr>
                                                    <th>{{ __('Service') }}</th>
                                                    <th>{{ __('Service Task') }}</th>
                                                    <th>{{ __('Task Duration') }}</th>
                                                    <th>{{ __('Description') }}</th>
                                                    <th>{{ __('Status') }}</th>
                                                    <th>#</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                @foreach ($workorder->tasks as $task)
                                                    <tr>
                                                        <td>{{ !empty($task->services) ? $task->services->title : '-' }}
                                                        </td>
                                                        <td>{{ $task->service_task }}</td>
                                                        <td>{{ $task->duration }}</td>
                                                        <td>{{ $task->description }}</td>
                                                        <td>
                                                            @if ($task->status == 'pending')
                                                                <span
                                                                    class="badge text-bg-warning">{{ \App\Models\WOServiceTask::$status[$task->status] }}</span>
                                                            @elseif($task->status == 'in_progress')
                                                                <span
                                                                    class="badge text-bg-primary">{{ \App\Models\WOServiceTask::$status[$task->status] }}</span>
                                                            @elseif($task->status == 'on_hold')
                                                                <span
                                                                    class="badge text-bg-danger">{{ \App\Models\WOServiceTask::$status[$task->status] }}</span>
                                                            @elseif($task->status == 'completed')
                                                                <span
                                                                    class="badge text-bg-success">{{ \App\Models\WOServiceTask::$status[$task->status] }}</span>
                                                            @endif
                                                        </td>
                                                        <td>
                                                            <div class="cart-action">
                                                                {!! Form::open(['method' => 'DELETE', 'route' => ['workorder.service.task.destroy', $workorder->id, $task->id]]) !!}

                                                                @can('edit workorder service task')
                                                                    <a class="avtar avtar-xs btn-link-secondary text-secondary customModal"
                                                                        data-bs-toggle="tooltip" data-size="md"
                                                                        data-bs-original-title="{{ __('Edit') }}"
                                                                        href="#"
                                                                        data-url="{{ route('workorder.service.task.edit', [$workorder->id, $task->id]) }}"
                                                                        data-title="{{ __('Edit Task') }}"> <i
                                                                            data-feather="edit"></i></a>
                                                                @endcan
                                                                @can('delete workorder service task')
                                                                    <a class=" avtar avtar-xs btn-link-danger text-danger confirm_dialog"
                                                                        data-bs-toggle="tooltip"
                                                                        data-bs-original-title="{{ __('Detete') }}"
                                                                        href="#"> <i data-feather="trash-2"></i></a>
                                                                @endcan
                                                                {!! Form::close() !!}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                @endforeach
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>

                        </div>



                        <div class="tab-pane" id="profile-3" role="tabpanel" aria-labelledby="profile-tab-3">
                            <div class="card border">
                                <div class="card-header">
                                    <div class="row align-items-center">
                                        <div class="col-sm-6">
                                            <h5 class="mb-0">{{ __('Service Appointment List') }}</h5>
                                        </div>
                                        <div class="col-sm-6 text-sm-end">
                                            @if (Gate::check('create service appointment'))
                                                <a class="btn btn-secondary btn-sm me-2 customModal " href="#"
                                                    data-url="{{ route('workorder.service.appointment', $workorder->id) }}"
                                                    data-size="md" data-title="{{ __('Service Appointment') }}">
                                                    {{ __('Service Appointment') }}
                                                </a>
                                            @endif
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body pt-0">
                                    <div class="dt-responsive table-responsive">
                                        <table class="table table-hover">
                                            <thead>
                                                <tr>
                                                    <th>{{ __('Start Date') }}</th>
                                                    <th>{{ __('Start Time') }}</th>
                                                    <th>{{ __('End Date') }}</th>
                                                    <th>{{ __('End Time') }}</th>
                                                    <th>{{ __('Status') }}</th>
                                                    <th>{{ __('Description') }}</th>
                                                    <th>#</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                @php $appointment=$workorder->appointments; @endphp
                                                @if (!empty($appointment))
                                                    <tr>
                                                        <td>{{ dateFormat($appointment->start_date) }}</td>
                                                        <td>{{ timeFormat($appointment->start_time) }}</td>
                                                        <td>{{ dateFormat($appointment->end_date) }}</td>
                                                        <td>{{ timeFormat($appointment->end_time) }}</td>

                                                        <td>
                                                            @if (in_array($appointment->status, ['pending', 'on_hold']))
                                                                <span
                                                                    class="badge text-bg-warning">{{ \App\Models\WOServiceAppointment::$status[$appointment->status] }}</span>
                                                            @elseif(in_array($appointment->status, ['schedule', 'reschedule']))
                                                                <span
                                                                    class="badge text-bg-primary">{{ \App\Models\WOServiceAppointment::$status[$appointment->status] }}</span>
                                                            @elseif($appointment->status == 'dispatched')
                                                                <span
                                                                    class="badge text-bg-info">{{ \App\Models\WOServiceAppointment::$status[$appointment->status] }}</span>
                                                            @elseif($appointment->status == 'completed')
                                                                <span
                                                                    class="badge text-bg-success">{{ \App\Models\WOServiceAppointment::$status[$appointment->status] }}</span>
                                                            @else
                                                                <span
                                                                    class="badge text-bg-danger">{{ \App\Models\WOServiceAppointment::$status[$appointment->status] }}</span>
                                                            @endif
                                                        </td>
                                                        <td>{{ !empty($appointment->notes) ? $appointment->notes : '-' }}
                                                        </td>
                                                        <td>
                                                            <div class="cart-action">
                                                                {!! Form::open(['method' => 'DELETE', 'route' => ['workorder.service.appointment.destroy', $workorder->id]]) !!}
                                                                @can('delete service appointment')
                                                                    <a class=" avtar avtar-xs btn-link-danger text-danger confirm_dialog"
                                                                        data-bs-toggle="tooltip"
                                                                        data-bs-original-title="{{ __('Detete') }}"
                                                                        href="#"> <i data-feather="trash-2"></i></a>
                                                                @endcan
                                                                {!! Form::close() !!}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                @endif
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- [ sample-page ] end -->
    </div>







@endsection
