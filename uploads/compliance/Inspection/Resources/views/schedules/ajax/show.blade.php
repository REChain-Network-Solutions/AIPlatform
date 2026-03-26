<style>
    #logo {
        height: 50px;
    }
</style>

<!-- INVOICE CARD START -->



<div class="card border-0 schedule">
    <!-- CARD BODY START -->
    <div class="card-body">

        @if ($message = Session::get('success'))
            <div class="alert alert-success alert-dismissable">
                <button type="button" class="close" data-dismiss="alert" aria-hidden="true"></button>
                <i class="fa fa-check"></i> {!! $message !!}
            </div>
            <?php Session::forget('success'); ?>
        @endif

        @if ($message = Session::get('error'))
            <div class="custom-alerts alert alert-danger fade in">
                <button type="button" class="close" data-dismiss="alert" aria-hidden="true"></button>
                {!! $message !!}
            </div>
            <?php Session::forget('error'); ?>
        @endif

        <div class="invoice-table-wrapper">
            <table width="100%">

                <tr class="inv-num">
                    <td align="right">
                        <table class="inv-num-date text-dark f-13 mt-3">
                            <tr>
                                <td class="bg-light-grey border-right-0 f-w-500">
                                    @lang('modules.inspection_schedules.scheduleNumber')</td>
                                <td class="border-left-0">{{ $schedule->subject }}</td>
                            </tr>

                            <tr>
                                <td class="bg-light-grey border-right-0 f-w-500">
                                    @lang('modules.inspection_schedules.scheduleDate')</td>
                                <td class="border-left-0">{{ $schedule->issue_date->translatedFormat(company()->date_format) }}
                                </td>
                            </tr>


                        </table>
                    </td>
                </tr>
                <tr>
                    <td height="20"></td>
                </tr>
            </table>
            <table width="100%">
                <tr class="inv-unpaid">
                    <td class="f-14 text-dark">
                        <p class="mb-0 text-left">



                        </p>
                    </td>


                </tr>
                <tr>
                    <td height="30" colspan="2"></td>
                </tr>
            </table>
            <table width="100%" class="inv-desc d-none d-lg-table d-md-table">
                <tr>
                    <td colspan="2">
                        <table class="inv-detail f-14 table-responsive-sm" width="100%">

                            @foreach ($schedule->items as $item)
                                @if ($item->type == 'item')
                                    <tr class="text-dark font-weight-semibold f-13">
                                        <td>{{ ucfirst($item->item_name) }}</td>

                                    </tr>

                                @endif
                            @endforeach

                            <tr>

                            </tr>
                        </table>
                    </td>

                </tr>
            </table>
            <table width="100%" class="inv-desc-mob d-block d-lg-none d-md-none">

                @foreach ($schedule->items as $item)
                    @if ($item->type == 'item')
                        <tr>
                            <th width="50%" class="bg-light-grey text-dark-grey font-weight-bold">
                                @lang('app.description')</th>
                            <td class="p-0 ">
                                <table>
                                    <tr width="100%" class="font-weight-semibold f-13">
                                        <td class="border-left-0 border-right-0 border-top-0">
                                            {{ ucfirst($item->item_name) }}</td>
                                    </tr>

                                </table>
                            </td>
                        </tr>
                        <tr>
                            <th width="50%" class="bg-light-grey text-dark-grey font-weight-bold">
                                {{$schedule->unit ? $schedule->unit->unit_type: 'Qty\hrs'}}</th>
                            <td width="50%">{{ $item->quantity }}</td>
                        </tr>
                        <tr>
                            <th width="50%" class="bg-light-grey text-dark-grey font-weight-bold">
                                @lang('modules.inspection_schedules.unitPrice')
                                ({{ $schedule->currency->currency_code }})</th>
                            <td width="50%">{{ currency_format($item->unit_price, $schedule->currency_id, false) }}
                            </td>
                        </tr>
                        <tr>
                            <th width="50%" class="bg-light-grey text-dark-grey font-weight-bold">
                                @lang('modules.inspection_schedules.amount')
                                ({{ $schedule->currency->currency_code }})</th>
                            <td width="50%">{{ currency_format($item->amount, $schedule->currency_id, false) }}</td>
                        </tr>
                        <tr>
                            <td height="3" class="p-0 " colspan="2"></td>
                        </tr>
                    @endif
                @endforeach







            </table>
            <table class="inv-note">
                <tr>
                    <td height="30" colspan="2"></td>
                </tr>
                <tr>
                    <td>
                        <table>
                            <tr>@lang('app.note')</tr>
                            <tr>
                                <p class="text-dark-grey">{!! !empty($schedule->note) ? $schedule->note : '--' !!}</p>
                            </tr>
                        </table>
                    </td>

                </tr>
                <tr>
                    <td>
                        <table>
                            <tr>

                            </tr>
                        </table>
                    </td>
                </tr>



            </table>
        </div>
    </div>
    <!-- CARD BODY END -->
    <!-- CARD FOOTER START -->
    <div class="card-footer bg-white border-0 d-flex justify-content-start py-0 py-lg-4 py-md-4 mb-4 mb-lg-3 mb-md-3 ">

        <div class="d-flex">
            <div class="inv-action mr-3 mr-lg-3 mr-md-3 dropup">
                <button class="dropdown-toggle btn-primary" type="button" id="dropdownMenuButton"
                    data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">@lang('app.action')
                    <span><i class="fa fa-chevron-up f-15"></i></span>
                </button>
                <!-- DROPDOWN - INFORMATION -->
                <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton" tabindex="0">

                    @if ($schedule->status == 'paid' && !in_array('client', user_roles()) && $schedule->amountPaid() == 0)
                        <li>
                            <a class="dropdown-item f-14 text-dark"
                                href="{{ route('inspection_schedules.edit', [$schedule->id]) }}">
                                <i class="fa fa-edit f-w-500 mr-2 f-11"></i> @lang('app.edit')
                            </a>
                        </li>
                    @endif





                    <li>

                    </li>

                    @if ($schedule->status != 'canceled' && !$schedule->credit_note && !in_array('client', user_roles()))
                        <li>
                            <a class="dropdown-item f-14 text-dark sendButton" href="javascript:;"
                                data-schedule-id="{{ $schedule->id }}">
                                <i class="fa fa-paper-plane f-w-500 mr-2 f-11"></i> @lang('app.send')
                            </a>
                        </li>
                    @endif











                    @if (!in_array($schedule->status, ['canceled', 'draft']) && !$schedule->credit_note && $schedule->send_status)
                        <li>
                            <a class="dropdown-item f-14 text-dark btn-copy" href="javascript:;"
                                data-clipboard-text="{{ route('front.schedule', $schedule->hash) }}">
                                <i class="fa fa-copy f-w-500  mr-2 f-12"></i>
                                @lang('modules.inspection_schedules.copyPaymentLink')
                            </a>
                        </li>
                        <li>
                            <a class="dropdown-item f-14 text-dark"
                                href="{{ route('front.schedule', $schedule->hash) }}" target="_blank">
                                <i class="fa fa-external-link-alt f-w-500  mr-2 f-12"></i>
                                @lang('modules.payments.paymentLink')
                            </a>
                        </li>
                    @endif

                    @if ($addSchedulesPermission == 'all' || $addSchedulesPermission == 'added')
                        <a href="{{ route('inspection_schedules.create') . '?schedule=' . $schedule->id }}"
                            class="dropdown-item"><i class="fa fa-copy mr-2"></i> @lang('app.create')
                            @lang('app.duplicate')</a>
                    @endif




                </ul>
            </div>



            <x-forms.button-cancel :link="route('inspection_schedules.index')" class="border-0 mr-3">@lang('app.cancel')
            </x-forms.button-cancel>

        </div>


    </div>
    <!-- CARD FOOTER END -->

</div>
<!-- INVOICE CARD END -->

{{-- Custom fields data --}}
@if (isset($fields) && count($fields) > 0)
    <div class="row mt-4">
        <!-- TASK STATUS START -->
        <div class="col-md-12">
            <x-cards.data>
                <x-forms.custom-field-show :fields="$fields" :model="$schedule"></x-forms.custom-field-show>
            </x-cards.data>
        </div>
    </div>
@endif

<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script src="{{ asset('vendor/jquery/clipboard.min.js') }}"></script>

<script>
    var clipboard = new ClipboardJS('.btn-copy');

    clipboard.on('success', function(e) {
        Swal.fire({
            icon: 'success',
            text: '@lang('app.copied')',
            toast: true,
            position: 'top-end',
            timer: 3000,
            timerProgressBar: true,
            showConfirmButton: false,
            customClass: {
                confirmButton: 'btn btn-primary',
            },
            showClass: {
                popup: 'swal2-noanimation',
                backdrop: 'swal2-noanimation'
            },
        })
    });











    $('body').on('click', '.cancel-schedule', function() {
        var id = $(this).data('schedule-id');
        Swal.fire({
            title: "@lang('messages.sweetAlertTitle')",
            text: "@lang('messages.scheduleText')",
            icon: 'warning',
            showCancelButton: true,
            focusConfirm: false,
            confirmButtonText: "@lang('app.yes')",
            cancelButtonText: "@lang('app.cancel')",
            customClass: {
                confirmButton: 'btn btn-primary mr-3',
                cancelButton: 'btn btn-secondary'
            },
            showClass: {
                popup: 'swal2-noanimation',
                backdrop: 'swal2-noanimation'
            },
            buttonsStyling: false
        }).then((result) => {
            if (result.isConfirmed) {
                var token = "{{ csrf_token() }}";

                var url = "{{ route('inspection_schedules.update_status', ':id') }}";
                url = url.replace(':id', id);

                $.easyAjax({
                    type: 'GET',
                    url: url,
                    container: '#inspection_schedules-table',
                    blockUI: true,
                    success: function(response) {
                        if (response.status == "success") {
                            window.location.reload();
                        }
                    }
                });
            }
        });
    });

    $('body').on('click', '.delete-schedule', function() {
        var id = $(this).data('schedule-id');
        Swal.fire({
            title: "@lang('messages.sweetAlertTitle')",
            text: "@lang('messages.recoverRecord')",
            icon: 'warning',
            showCancelButton: true,
            focusConfirm: false,
            confirmButtonText: "@lang('messages.confirmDelete')",
            cancelButtonText: "@lang('app.cancel')",
            customClass: {
                confirmButton: 'btn btn-primary mr-3',
                cancelButton: 'btn btn-secondary'
            },
            showClass: {
                popup: 'swal2-noanimation',
                backdrop: 'swal2-noanimation'
            },
            buttonsStyling: false
        }).then((result) => {
            if (result.isConfirmed) {
                var token = "{{ csrf_token() }}";

                var url = "{{ route('inspection_schedules.destroy', ':id') }}";
                url = url.replace(':id', id);

                $.easyAjax({
                    type: 'POST',
                    url: url,
                    blockUI: true,
                    data: {
                        '_token': token,
                        '_method': 'DELETE'
                    },
                    success: function(response) {
                        if (response.status == "success") {
                            window.location.href = "{{ route('inspection_schedules.index') }}";
                        }
                    }
                });
            }
        });
    });




</script>
