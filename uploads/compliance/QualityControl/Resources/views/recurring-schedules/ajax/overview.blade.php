<style>
    #logo {
        height: 50px;
    }

</style>

<!-- INVOICE CARD START -->
{{-- @if (!is_null($schedule->project) && !is_null($schedule->project->client) && !is_null($schedule->project->client->clientDetails))
    @php
        $client = $schedule->project->client;
    @endphp
@elseif(!is_null($schedule->client_id) && !is_null($schedule->clientDetails))
    @php
        $client = $schedule->client;
    @endphp
@endif --}}

<div class="d-lg-flex">
    <div class="w-100 py-0 py-md-0 ">
        <x-cards.data :title="__('app.recurring') . ' ' . __('app.details')" class=" mt-4">
            <div class="col-12 px-0 pb-3 d-block d-lg-flex d-md-flex">
                <p class="mb-0 text-lightest f-14 w-30 d-inline-block text-capitalize">
                    Completed/Total</p>
                <p class="mb-0 text-dark-grey f-14 ">
                    @if(!is_null($schedule->billing_cycle))
                        {{$schedule->recurrings->count()}}/{{$schedule->billing_cycle}}
                    @else
                        {{$schedule->recurrings->count()}}/<span class="px-1"><label class="badge badge-primary">@lang('app.infinite')</label></span>
                    @endif
                </p>
            </div>
            @if (count($schedule->recurrings) > 0)
                <x-cards.data-row :label="__('First Schedule Date')"
                :value="$schedule->recurrings->last()->issue_date->translatedFormat(company()->date_format)" />
            @else
                <x-cards.data-row :label="__('First Schedule Date')"
                :value="$schedule->issue_date ? $schedule->issue_date->translatedFormat(company()->date_format) : '----'" />
            @endif

            <x-cards.data-row :label="__('Next Schedule').' '.__('app.date')"
            :value="$schedule->next_schedule_date ? $schedule->next_schedule_date->translatedFormat(company()->date_format) : '----'" />
        </x-cards.data>
    </div>
</div>
<div class="d-lg-flex">

    <div class="w-100 py-0 py-lg-4 py-md-0 ">

        <div class="card border-0 schedule">
            <!-- CARD BODY START -->
            <div class="card-body">

                <div class="invoice-table-wrapper">
                    <table width="100%" class="">
                        <tr class="inv-logo-heading">
                            <td><img src="{{ invoice_setting()->logo_url }}"
                                    alt="{{ mb_ucwords(company()->company_name) }}" id="logo" /></td>
                            <td align="right"
                                class="font-weight-bold f-21 text-dark text-uppercase mt-4 mt-lg-0 mt-md-0">
                                Schedule Job</td>
                        </tr>
                        <tr class="inv-num">
                            <td class="f-14 text-dark">
                                <p class="mt-3 mb-0">
                                    {{ mb_ucwords(company()->company_name) }}<br>
                                    @if (!is_null($settings))
                                        {!! nl2br(default_address()->address) !!}<br>
                                        {{ company()->company_phone }}
                                    @endif
                                    {{-- @if ($scheduleSetting->show_gst == 'yes' && $schedule->address)
                                        <br>{{ $schedule->address->tax_name }}: {{ $schedule->address->tax_number }}
                                    @endif --}}
                                </p><br>
                            </td>
                            <td align="right">
                                <table class="inv-num-date text-dark f-13 mt-3">
                                    <tr>
                                        <td class="bg-light-grey border-right-0 f-w-500">
                                            Schedule Date</td>
                                        <td class="border-left-0">
                                            {{ $schedule->issue_date->translatedFormat(company()->date_format) }}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td class="bg-light-grey border-right-0 f-w-500">Next Schedule Date</td>
                                        <td class="border-left-0">
                                            {{ $schedule->next_schedule_date->translatedFormat(company()->date_format) }}
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
                                <p class="mb-0 text-left"><span
                                        class="text-dark-grey text-capitalize"></span><br>

                                    {{-- @php
                                        if ($schedule->project && $schedule->project->client) {
                                            $client = $schedule->project->client;
                                        } elseif ($schedule->client_id != '') {
                                            $client = $schedule->client;
                                        } elseif ($schedule->estimate && $schedule->estimate->client) {
                                            $client = $schedule->estimate->client;
                                        }
                                    @endphp --}}

                                    {{-- {{ mb_ucwords($client->name) }}<br>
                                    {{ mb_ucwords($client->clientDetails->company_name) }}<br>
                                    {!! nl2br($client->clientDetails->address) !!}</p> --}}
                            </td>
                            @if ($schedule->shipping_address)
                                <td class="f-14 text-black">
                                    <p class="mb-0 text-left"><span
                                            class="text-dark-grey text-capitalize">@lang('app.shippingAddress')</span><br>
                                        {!! nl2br($client->clientDetails->address) !!}</p>
                                </td>
                            @endif
                        </tr>
                        <tr>
                            <td height="30" colspan="2"></td>
                        </tr>
                    </table>
                    <table width="100%" class="inv-desc d-none d-lg-table d-md-table">
                        <tr>
                            <td colspan="2">
                                <table class="inv-detail f-14 table-responsive-sm" width="100%">
                                    <tr class="i-d-heading bg-light-grey text-dark-grey font-weight-bold">
                                        <td class="border-right-0">Standar Bersih</td>
                                        {{-- @if ($scheduleSetting->hsn_sac_code_show)
                                            <td class="border-right-0 border-left-0" align="right">@lang('app.hsnSac')
                                        @endif --}}
                                        <td class="border-right-0 border-left-0" align="right">
                                        </td>

                                    </tr>
                                    @foreach ($schedule->items as $item)

                                            <tr class="text-dark">
                                                <td>{{ ucfirst($item->item_name) }}</td>

                                            </tr>
                                            {{-- @if ($item->item_summary != '' || $item->recurringScheduleItemImage)
                                                <tr class="text-dark">
                                                    <td colspan="{{ $scheduleSetting->hsn_sac_code_show ? '5' : '4' }}"
                                                        class="border-bottom-0">{!! nl2br(strip_tags($item->item_summary)) !!}
                                                        @if ($item->recurringScheduleItemImage)
                                                            <p class="mt-2">
                                                                <a href="javascript:;" class="img-lightbox"
                                                                    data-image-url="{{ $item->recurringScheduleItemImage->file_url }}">
                                                                    <img src="{{ $item->recurringScheduleItemImage->file_url }}"
                                                                        width="80" height="80" class="img-thumbnail">
                                                                </a>
                                                            </p>
                                                        @endif
                                                    </td>
                                                </tr>
                                            @endif --}}

                                    @endforeach
                                    <p class="mb-0">
                                        {{-- @if ($scheduleSetting->show_project == 1 && isset($schedule->project->project_name))
                                            <span class="text-dark-grey text-capitalize">@lang('modules.inspection_schedules.projectName')</span><br>
                                            {{ $schedule->project->project_name }}
                                        @endif --}}
                                        <br><br>
                                    </p>
                                    <tr>
                                        <td colspan="2" class="blank-td border-bottom-0 border-left-0 border-right-0">
                                        </td>
                                        <td colspan="3" class="p-0 ">
                                            <table width="100%">
                                                {{-- <tr class="text-dark-grey" align="right">
                                                    <td class="w-50 border-top-0 border-left-0">
                                                        @lang('modules.inspection_schedules.subTotal')</td>
                                                    <td class="border-top-0 border-right-0">
                                                        {{ number_format((float) $schedule->sub_total, 2, '.', '') }}
                                                    </td>
                                                </tr>
                                                @if ($discount != 0 && $discount != '')
                                                    <tr class="text-dark-grey" align="right">
                                                        <td class="w-50 border-top-0 border-left-0">
                                                            @lang('modules.inspection_schedules.discount')</td>
                                                        <td class="border-top-0 border-right-0">
                                                            {{ number_format((float) $discount, 2, '.', '') }}</td>
                                                    </tr>
                                                @endif
                                                @foreach ($taxes as $key => $tax)
                                                    <tr class="text-dark-grey" align="right">
                                                        <td class="w-50 border-top-0 border-left-0">
                                                            {{ mb_strtoupper($key) }}</td>
                                                        <td class="border-top-0 border-right-0">
                                                            {{ number_format((float) $tax, 2, '.', '') }}</td>
                                                    </tr>
                                                @endforeach
                                                <tr class=" text-dark-grey font-weight-bold" align="right">
                                                    <td class="w-50 border-bottom-0 border-left-0">
                                                        @lang('modules.inspection_schedules.total')</td>
                                                    <td class="border-bottom-0 border-right-0">
                                                        {{ number_format((float) $schedule->total, 2, '.', '') }}</td>
                                                </tr> --}}

                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </td>

                        </tr>
                    </table>
                    <table width="100%" class="inv-desc-mob d-block d-lg-none d-md-none">

                        @foreach ($schedule->items as $item)

                                <tr>
                                    <th width="50%" class="bg-light-grey text-dark-grey font-weight-bold">
                                        @lang('app.description')</th>
                                    <td class="p-0 ">
                                        <table>
                                            <tr width="100%">
                                                <td class="border-left-0 border-right-0 border-top-0">
                                                    {{ ucfirst($item->item_name) }}</td>
                                            </tr>
                                            {{-- @if ($item->item_summary != '')
                                                <tr>
                                                    <td class="border-left-0 border-right-0 border-bottom-0">
                                                        {!! nl2br(strip_tags($item->item_summary)) !!}</td>
                                                </tr>
                                            @endif --}}
                                        </table>
                                    </td>
                                </tr>



                        @endforeach


                    </table>

                </div>
            </div>
            <!-- CARD BODY END -->

        </div>

    </div>
</div>
<!-- INVOICE CARD END -->
