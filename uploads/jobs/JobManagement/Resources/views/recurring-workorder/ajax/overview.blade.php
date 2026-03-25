<style>
    #logo {
        height: 50px;
    }
</style>

<div class="d-lg-flex">
    <div class="w-100 py-0 py-md-0 ">
        <x-cards.data :title="__('app.recurring') . ' ' . __('app.details')" class=" mt-4">
            <div class="col-12 px-0 pb-3 d-block d-lg-flex d-md-flex">
                <p class="mb-0 text-lightest f-14 w-30 d-inline-block text-capitalize">
                    Completed/Total</p>
                <p class="mb-0 text-dark-grey f-14 ">
                    @if (!is_null($schedule->billing_cycle))
                        {{ $schedule->recurrings->count() }}/{{ $schedule->billing_cycle }}
                    @else
                        {{ $schedule->recurrings->count() }}/<span class="px-1"><label
                                class="badge badge-primary">@lang('app.infinite')</label></span>
                    @endif
                </p>
            </div>
            @if (count($schedule->recurrings) > 0)
                <x-cards.data-row :label="__('engineerings::app.menu.firstSchedule')" :value="Carbon\Carbon::parse($schedule->recurrings->last()->issue_date)->translatedFormat(company()->date_format)" />
            @else
                <x-cards.data-row :label="__('engineerings::app.menu.firstSchedule')" :value="$schedule->issue_date
                    ? $schedule->issue_date->translatedFormat(company()->date_format)
                    : '----'" />
            @endif

            <x-cards.data-row :label="__('engineerings::app.menu.nextSchedule')" :value="$schedule->next_schedule_date
                ? $schedule->next_schedule_date->translatedFormat(company()->date_format)
                : '----'" />
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
                               @lang('engineerings::modules.recWorkOrders')</td>
                        </tr>
                        <tr class="inv-num">
                            <td class="f-14 text-dark">
                                <p class="mt-3 mb-0">
                                    {{ mb_ucwords(company()->company_name) }}<br>
                                    @if (!is_null($settings))
                                        {!! nl2br(company()->address) !!}<br>
                                        {{-- {!! nl2br(default_address()->address) !!}<br> --}}
                                        {{ company()->company_phone }}
                                    @endif
                                </p><br>
                            </td>
                            <td align="right">
                                <table class="inv-num-date text-dark f-13 mt-3">
                                    <tr>
                                        <td class="bg-light-grey border-right-0 f-w-500">
                                            @lang('engineerings::app.menu.scheduleStart')</td>
                                        <td class="border-left-0">
                                            {{ $schedule->issue_date->translatedFormat(company()->date_format) }}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td class="bg-light-grey border-right-0 f-w-500">@lang('engineerings::app.menu.nextSchedule')</td>
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
                        <tr>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.WRid')</span><br>
                                    {{ $schedule->wr->wr_no }}
                                </p>
                            </td>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.unit')</span><br>
                                    {{ ucwords($schedule->unit->unit_name ) }}
                                </p>
                            </td>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.assets')</span><br>
                                    {{ ucwords($schedule->assets->type->type_name ) }}
                                </p>
                            </td>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.category')</span><br>
                                    {{ ucwords($schedule->category ) }}
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.priority')</span><br>
                                    {{ ucwords($schedule->priority ) }}
                                </p>
                            </td>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.estimateHours')</span><br>
                                    {{ $schedule->estimate_hours.':'.$schedule->estimate_minutes }}
                                </p>
                            </td>
                            <td class="f-14 text-black">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.scheduleStart')</span><br>
                                    {{ $schedule->schedule_start }}
                                </p>
                            </td>
                            <td class="f-14 text-black mb-3">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.scheduleFinish')</span><br>
                                    {{ $schedule->schedule_finish }}
                                </p>
                            </td>
                        </tr>
                        <tr class="mt-3">
                            <td class="f-14 text-black border p-3 mt-3" colspan="4">
                                <p class="mb-0 text-left">
                                    <span class="text-dark-grey text-capitalize">@lang('engineerings::app.menu.workDesc')</span><br>
                                    {{ ucwords($schedule->work_description) }}
                                </p>
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
            <!-- CARD BODY END -->

        </div>
    </div>
</div>
