<!doctype html>
<html lang="en">

<head>
    <!-- Required meta tags -->
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    {{-- <title>@lang('Form Request') - {{ $wo->name }}</title> --}}
    <meta name="msapplication-TileColor" content="#ffffff">
    <meta name="msapplication-TileImage" content="">
    {{-- {{ $company->favicon_url }} --}}
    <meta name="theme-color" content="#ffffff">

    <style>
        @font-face {
            font-family: 'THSarabunNew';
            font-style: normal;
            font-weight: normal;
            src: url("{{ storage_path('fonts/THSarabunNew.ttf') }}") format('truetype');
        }

        @font-face {
            font-family: 'THSarabunNew';
            font-style: normal;
            font-weight: bold;
            src: url("{{ storage_path('fonts/THSarabunNew_Bold.ttf') }}") format('truetype');
        }

        @font-face {
            font-family: 'THSarabunNew';
            font-style: italic;
            font-weight: bold;
            src: url("{{ storage_path('fonts/THSarabunNew_Bold_Italic.ttf') }}") format('truetype');
        }

        @font-face {
            font-family: 'THSarabunNew';
            font-style: italic;
            font-weight: bold;
            src: url("{{ storage_path('fonts/THSarabunNew_Italic.ttf') }}") format('truetype');
        }

        * {
            font-family: Verdana, DejaVu Sans, sans-serif;
        }


        .bg-grey {
            background-color: #F2F4F7;
        }

        .bg-white {
            background-color: #fff;
        }

        .border-radius-25 {
            border-radius: 0.25rem;
        }

        .p-25 {
            padding: 1.25rem;
        }

        .f-11 {
            font-size: 11px;
        }

        .f-13 {
            font-size: 13px;
        }

        .f-14 {
            font-size: 13px;
        }

        .f-15 {
            font-size: 13px;
        }

        .f-21 {
            font-size: 17px;
        }

        .text-black {
            color: #28313c;
        }

        .text-grey {
            color: #616e80;
        }

        .font-weight-700 {
            font-weight: 700;
        }

        .text-uppercase {
            text-transform: uppercase;
        }

        .text-capitalize {
            text-transform: capitalize;
        }

        .line-height {
            line-height: 20px;
        }

        .mt-1 {
            margin-top: 1rem;
        }

        .mb-0 {
            margin-bottom: 0px;
        }

        .b-collapse {
            border-collapse: collapse;
        }

        .heading-table-left {
            padding: 6px;
            border: 1px solid #DBDBDB;
            font-weight: bold;
            background-color: #f1f1f3;
            border-right: 0;
        }

        .heading-table-right {
            padding: 6px;
            border: 1px solid #DBDBDB;
            border-left: 0;
        }

        .unpaid {
            color: #d30000;
            border: 1px solid #d30000;
            position: relative;
            padding: 5px 10px;
            font-size: 14px;
            border-radius: 0.25rem;
            width: 100px;
            text-align: center;
            margin-top: 50px;
        }

        .other {
            color: #000000;
            border: 1px solid #000000;
            position: relative;
            padding: 5px 10px;
            font-size: 14px;
            border-radius: 0.25rem;
            width: 120px;
            text-align: center;
            margin-top: 50px;
        }

        .paid {
            color: #28a745 !important;
            border: 1px solid #28a745;
            position: relative;
            padding: 6px 12px;
            font-size: 14px;
            border-radius: 0.25rem;
            width: 100px;
            text-align: center;
            margin-top: 50px;
        }

        .main-table-heading {
            border: 1px solid #DBDBDB;
            background-color: #f1f1f3;
            font-weight: 700;
        }

        .main-table-heading td {
            padding: 5px 8px;
            border: 1px solid #DBDBDB;
        }

        .main-table-items td {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
        }

        .total-box {
            border: 1px solid #e7e9eb;
            padding: 0px;
            border-bottom: 0px;
        }

        .subtotal {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
            border-top: 0;
            border-left: 0;
            border-right: 0;
        }

        .subtotal-amt {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
            border-top: 0;
            border-left: 0;
            border-right: 0;
        }

        .total {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
            border-top: 0;
            font-weight: 700;
            border-left: 0;
            border-right: 0;
        }

        .total-amt {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
            border-top: 0;
            border-left: 0;
            border-right: 0;
            font-weight: 700;
        }

        .balance {
            font-size: 14px;
            font-weight: bold;
            background-color: #f1f1f3;
        }

        .balance-left {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
            border-top: 0;
            border-left: 0;
            border-right: 0;
        }

        .balance-right {
            padding: 5px 8px;
            border: 1px solid #e7e9eb;
            border-top: 0;
            border-left: 0;
            border-right: 0;
        }

        .centered {
            margin: 0 auto;
        }

        .rightaligned {
            margin-right: 0;
            margin-left: auto;
        }

        .leftaligned {
            margin-left: 0;
            margin-right: auto;
        }

        .page_break {
            page-break-before: always;
        }

        #logo {
            height: 50px;
        }

        .word-break {
            max-width: 175px;
            word-wrap: break-word;
        }

        .summary {
            padding: 11px 10px;
            border: 1px solid #e7e9eb;
            font-size: 11px;
        }

        .border-left-0 {
            border-left: 0 !important;
        }

        .border-right-0 {
            border-right: 0 !important;
        }

        .border-top-0 {
            border-top: 0 !important;
        }

        .border-bottom-0 {
            border-bottom: 0 !important;
        }

        .h3-border {
            border-bottom: 1px solid #AAAAAA;
        }

        @if ($invoiceSetting->locale == 'th')

            table td {
                font-weight: bold !important;
                font-size: 20px !important;
            }

            .description {
                font-weight: bold !important;
                font-size: 16px !important;
            }
        @endif
    </style>
</head>

<body class="content-wrapper">
    <table class="bg-white" border="0" cellpadding="0" cellspacing="0" width="100%" role="presentation">
        <tbody>
            <!-- Table Row Start -->
            <tr class="mb-2">
                <td><img src="{{ $invoiceSetting->logo_url }}" alt="{{ mb_ucwords($company->company_name) }}"
                        id="logo" /></td>

                <td align="right" class="f-21 text-black font-weight-700 text-uppercase">@lang('engineerings::modules.wo')<br>
                    <table class="text-black mt-1 f-13 b-collapse rightaligned">
                        <tr>
                            <td class="heading-table-left mb-2">@lang('engineerings::app.menu.noWO')</td>
                            <td class="heading-table-right mb-2">{{ ucwords($wo->nomor_wo) ?? '--' }}</td>
                        </tr>
                        <tr>
                            <td class="heading-table-left mb-2">@lang('engineerings::app.menu.WRid')</td>
                            <td class="heading-table-right mb-2">{{ ucwords($wo->wr->wr_no) ?? '--' }}</td>
                        </tr>
                    </table>
                </td>
            </tr>
            <p class="line-height mb-0"></p>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize"> @lang('engineerings::app.menu.ticketSubject')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">: {{ ucwords($wo->ticket->subject) }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.area')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords(optional(optional($wo->house)->area)->area_name ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.houses')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">: {{ mb_ucwords(optional($wo->house)->house_name ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.assets')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords(optional(optional($wo->assets)->type)->type_name ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.category')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->category ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.priority')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->priority ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.status')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->status ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.scheduleStart')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->schedule_start ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.scheduleFinish')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->schedule_finish ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.estimateHours')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords(($wo->estimate_hours ?? '--') . ' hours,' . ($wo->estimate_minutes ?? '--')) }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.actualStart')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ $wo->actual_start ?? '--' }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.actualFinish')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ $wo->actual_finish ?? '--' }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.actualHours')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ ($wo->actual_hours ?? '--') . ' hours,' . ($wo->actual_minutes ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.workDesc')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->work_description ?? '--') }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.completitionNotes')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ mb_ucwords($wo->completion_notes ?? '--') }}</span>
                </td>
            </tr>
            <!-- Table Row End -->
        </tbody>
    </table>

</body>

</html>
