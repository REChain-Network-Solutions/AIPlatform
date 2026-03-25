<!doctype html>
<html lang="en">

<head>
    <!-- Required meta tags -->
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>@lang('Form Request') - {{ $wr->name }}</title>
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

                <td align="right" class="f-21 text-black font-weight-700 text-uppercase">@lang('engineerings::modules.wr')<br>
                    <table class="text-black mt-1 f-13 b-collapse rightaligned">
                        <tr>
                            <td class="heading-table-left mb-2">@lang('engineerings::app.menu.noWR')</td>
                            <td class="heading-table-right mb-2">{{ ucwords($wr->wr_no) }}</td>
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
                    <span class="text-black text-capitalize">: {{ ucwords($wr->ticket->subject) }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.assignTo')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">: {{ $wr->user->name ?? '--' }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.remark')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">: {{ ucwords($wr->remark) }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.tenant')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ $wr->charge_by_tenant === 1 ? 'Yes' : ($wr->charge_by_tenant === 0 ? 'No' : null) }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.area')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ ucwords($wr->house->area->area_name) ?? '--' }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.houses')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ ucwords($wr->house->house_name) ?? '--' }}</span>
                </td>
            </tr>
            <tr>
                <td class="f-14 text-black" width="20%">
                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.problem')</span>
                </td>
                <td class="f-14 text-black" width="80%">
                    <span class="text-black text-capitalize">:
                        {{ ucwords($wr->problem) }}</span>
                </td>
            </tr>
            <!-- Table Row End -->
        </tbody>
    </table>
    <table width="100%" class="f-14 b-collapse">
        <tr>
            <td height="10" colspan="2"></td>
        </tr>
        @foreach ($wr->items as $item)
            <!-- Table Row Start -->
            <tr class="main-table-heading text-grey">
                <td width="35%" class="border bg-light-grey">@lang('Items')</td>
                <td width="12%" class="border bg-light-grey" align="right">@lang('Qty')</td>
                <td width="16%" class="border bg-light-grey" align="right">@lang('Harga')</td>
                <td width="17%" class="border bg-light-grey" align="right">@lang('Tax')</td>
            </tr>
            <!-- Table Row End -->
            <tr class="main-table-items text-black">
                <td width="35%" class="border">{{ ucwords($item->item->name) }}</td>
                <td width="12%" class="border" align="right"> {{ $item->qty }} <input type="hidden"
                        name="item_name" value="{{ $item->qty }}"></td>
                <td width="16%" class="border" align="right"> {{ $item->harga }} <input type="hidden"
                        name="harga" value="{{ $item->harga }}"></td>
                <td width="17%" class="border" align="right">
                    @if ($item->tax != 0)
                        {{ $item->tax . '%' }}
                    @endif
                    <input type="hidden" name="tax" value="{{ $item->tax }}">
                </td>
            </tr>
        @endforeach
        
        @foreach ($wr->services as $service)
            <!-- Table Row Start -->
            <tr class="main-table-heading text-grey">
                <td width="35%" class="border bg-light-grey">@lang('Service')</td>
                <td width="12%" class="border bg-light-grey" align="right">@lang('Qty')</td>
                <td width="16%" class="border bg-light-grey" align="right">@lang('Harga')</td>
                <td width="17%" class="border bg-light-grey" align="right">@lang('Tax')</td>
            </tr>
            <!-- Table Row End -->
            <tr class="main-table-items text-black">
                <td width="35%" class="border">{{ ucwords($service->service->name) }}</td>
                <td width="12%" class="border" align="right"> {{ $service->qty }} <input type="hidden"
                        name="item_name" value="{{ $service->qty }}"></td>
                <td width="16%" class="border" align="right"> {{ $service->harga }} <input type="hidden"
                        name="harga" value="{{ $service->harga }}"></td>
                <td width="17%" class="border" align="right">
                    @if ($service->tax != 0)
                        {{ $service->tax . '%' }}
                    @endif
                    <input type="hidden" name="tax" value="{{ $service->tax }}">
                </td>
            </tr>
        @endforeach
    </table>

</body>

</html>
