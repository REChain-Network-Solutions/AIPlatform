<!doctype html>
<html lang="en">

<head>
    <!-- Required meta tags -->
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>@lang('app.invoice') - {{ $invoice->invoice_number }}</title>
    <meta name="msapplication-TileColor" content="#ffffff">
    <meta name="msapplication-TileImage" content="{{ $company->favicon_url }}">
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

        @if($invoiceSetting->is_chinese_lang)
@font-face {
            font-family: SimHei;
            /*font-style: normal;*/
            font-weight: bold;
            src: url('{{ asset('fonts/simhei.ttf') }}') format('truetype');
        }

        @endif

    @php
        $font = '';
        if($invoiceSetting->locale == 'ja') {
            $font = 'ipag';
        } else if($invoiceSetting->locale == 'hi') {
            $font = 'hindi';
        } else if($invoiceSetting->locale == 'th') {
            $font = 'THSarabunNew';
        } else if($invoiceSetting->is_chinese_lang) {
            $font = 'SimHei';
        }else {
            $font = 'Verdana';
        }
    @endphp

    @if($invoiceSetting->is_chinese_lang)
        body {
            font-weight: normal !important;
        }

        @endif
    * {
            font-family: {{$font}}, DejaVu Sans, sans-serif;
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

        @if($invoiceSetting->locale == 'th')

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
    <tr>
        <td><img src="{{ $invoiceSetting->logo_url }}" alt="{{ mb_ucwords($company->company_name) }}"
                 id="logo"/></td>
        <td align="right" class="f-21 text-black font-weight-700 text-uppercase">@lang('app.invoice')<br>
            <table class="text-black mt-1 f-13 b-collapse rightaligned">
                <tr>
                    <td class="heading-table-left">@lang('modules.invoices.invoiceNumber')</td>
                    <td class="heading-table-right">{{ $invoice->invoice_number }}</td>
                </tr>
                @if ($creditNote)
                    <tr>
                        <td class="heading-table-left">@lang('app.credit-note')</td>
                        <td class="heading-table-right">{{ $creditNote->cn_number }}</td>
                    </tr>
                @endif
                <tr>
                    <td class="heading-table-left">@lang('modules.invoices.invoiceDate')</td>
                    <td class="heading-table-right">
                        {{ $invoice->issue_date->translatedFormat($invoice->company->date_format) }}
                    </td>
                </tr>
                @if (empty($invoice->order_id) && $invoice->status === 'unpaid' && $invoice->due_date->year > 1)
                    <tr>
                        <td class="heading-table-left">@lang('app.dueDate')</td>
                        <td class="heading-table-right">
                            {{ $invoice->due_date->translatedFormat($invoice->company->date_format) }}
                        </td>
                    </tr>
                @endif
            </table>
        </td>
    </tr>
    <!-- Table Row End -->
    <!-- Table Row Start -->
    <tr>
        <td class="f-14 text-black">
            <p class="line-height mb-0 ">
                <span class="text-grey text-capitalize">@lang('modules.invoices.billedFrom')</span><br>
                {{ mb_ucwords($company->company_name) }}<br>
                @if ($company->company_email)
                    {{ $company->company_email }}<br>
                @endif

                @if ($company->company_phone)
                    {{ $company->company_phone }}<br>
                @endif

                @if (!is_null($company) && $invoice->address)
                    {!! nl2br($invoice->address->address) !!}<br>
                @endif

                @if ($invoiceSetting->show_gst == 'yes' && $invoice->address)
                    {{ $invoice->address->tax_name }}: {{ $invoice->address->tax_number }}
                @endif
            </p>
            @if ($invoiceSetting->show_project == 1 && isset($invoice->project->project_name))
                <br>
                <p class="line-height mb-0"></p>
            @endif
        </td>
        <td class="f-14 text-black" align="right">
            @if (!is_null($invoice->project) && !is_null($invoice->project->client) && !is_null($invoice->project->client->clientDetails))
                @php
                    $client = $invoice->project->client;
                @endphp
            @elseif(!is_null($invoice->client_id) && !is_null($invoice->clientDetails))
                @php
                    $client = $invoice->client;
                @endphp
            @endif

            @if (($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes') && $client)
                <p class="line-height mb-0">
                            <span class="text-grey text-capitalize">
                                @lang('modules.invoices.billedTo')</span><br>

                    @if ($client->name && $invoiceSetting->show_client_name == 'yes')
                        {{ mb_ucwords($client->name) }}<br>
                    @endif

                    @if ($client->email && $invoiceSetting->show_client_email == 'yes')
                        {{ $client->email }}<br>
                    @endif

                    @if ($client->mobile && $invoiceSetting->show_client_phone == 'yes')
                        {{ $client->mobile }}<br>
                    @endif

                    @if ($client->clientDetails->company_name && $invoiceSetting->show_client_company_name == 'yes')
                        {{ mb_ucwords($client->clientDetails->company_name) }}<br>
                    @endif

                    @if ($client->clientDetails->address && $invoiceSetting->show_client_company_address == 'yes')
                        {!! nl2br($client->clientDetails->address) !!}
                    @endif

                    @if ($invoiceSetting->show_gst == 'yes' && !is_null($client->clientDetails->gst_number))
                        <br>@lang('app.gstIn'):
                        {{ $client->clientDetails->gst_number }}
                    @endif
                </p>
            @endif

            @if ($invoiceSetting->show_project == 1 && isset($invoice->project->project_name))
                <br>
                <p class="line-height mb-0">
                    <span class="text-grey text-capitalize">@lang('modules.invoices.projectName')</span>:
                    {{ $invoice->project->project_name }}
                </p>
            @endif

        </td>
    </tr>
    <!-- Table Row End -->
    <!-- Table Row Start -->
    <tr>
        <td height="10"></td>
    </tr>
    <!-- Table Row End -->
    <!-- Table Row Start -->
    <tr>
        <td colspan="2">
            <table border="0" cellpadding="0" cellspacing="0" width="100%">
                <tr>
                    <td class="f-14 text-black">
                        @if ($invoice->show_shipping_address == 'yes')
                            <p class="line-height"><span
                                    class="text-grey text-capitalize">@lang('app.shippingAddress')</span><br>
                                {!! nl2br($client->clientDetails->shipping_address) !!}</p>
                        @endif
                    </td>
                    @if ($invoiceSetting->show_status)
                        <td align="right">
                            <div style="margin: 0 0 auto auto"
                                 class="text-uppercase bg-white {{$invoice->status =='paid'|| $invoice->status=='unpaid'?$invoice->status:'other'}} rightaligned">
                                @if ($invoice->credit_note)
                                    @lang('app.credit-note')
                                @else
                                    @lang('modules.invoices.' . $invoice->status)
                                @endif
                            </div>
                        </td>
                    @endif
                </tr>
            </table>
        </td>
    </tr>
    </tbody>
</table>

<table width="100%" class="f-14 b-collapse">
    <tr>
        <td height="10" colspan="2"></td>
    </tr>
    <!-- Table Row Start -->
    <tr class="main-table-heading text-grey">
        <td width="40%">@lang('app.description')</td>
        @if ($invoiceSetting->hsn_sac_code_show)
            <td align="right">@lang('app.hsnSac')</td>
        @endif
        <td align="right">{{ isset($invoice->unit) ? $invoice->unit->unit_type : 'Qty\hrs' }}</td>
        <td align="right">@lang('modules.invoices.unitPrice')</td>
        <td align="right">@lang('modules.invoices.tax')</td>
        <td align="right"
            width="{{ $invoiceSetting->hsn_sac_code_show ? '17%' : '20%' }}">@lang('modules.invoices.amount')
            ({{ $invoice->currency->currency_code }})
        </td>
    </tr>
    <!-- Table Row End -->
@foreach ($invoice->items as $item)
    @if ($item->type == 'item')
        <!-- Table Row Start -->
            <tr class="main-table-items text-black">
                <td width="40%" class="border-bottom-0">
                    {{ ucfirst($item->item_name) }}
                </td>
                @if ($invoiceSetting->hsn_sac_code_show)
                    <td align="right" width="10%" class="border-bottom-0">
                        {{ $item->hsn_sac_code ?  : '--' }}</td>
                @endif
                <td align="right" width="10%" class="border-bottom-0">{{ $item->quantity }}</td>
                <td align="right"
                    class="border-bottom-0">{{ currency_format($item->unit_price, $invoice->currency_id, false) }}</td>
                <td align="right" class="border-bottom-0">{{ strtoupper($item->tax_list) }}</td>
                <td align="right" class="border-bottom-0"
                    width="{{ $invoiceSetting->hsn_sac_code_show ? '17%' : '20%' }}">
                    {{ currency_format($item->amount, $invoice->currency_id, false) }}</td>
            </tr>
            <!-- Table Row End -->
            @if ($item->item_summary != '' || $item->invoiceItemImage)
                {{-- DOMPDF HACK FOR RENDER IN TABLE --}}
</table>
<div class="f-13 summary text-black border-bottom-0 description">
    {!! nl2br(strip_tags($item->item_summary, ['p', 'b', 'strong', 'a'])) !!}
    @if ($item->invoiceItemImage)
        <p class="mt-2">
            <img src="{{ $item->invoiceItemImage->file_url }}" width="60" height="60" class="img-thumbnail">
        </p>
    @endif
</div>
{{-- DOMPDF HACK FOR RENDER IN TABLE --}}
<table class="bg-white" border="0" cellpadding="0" cellspacing="0" width="100%" role="presentation">
@endif
@endif
@endforeach
<!-- Table Row Start -->
    <tr>
        <td class="total-box" align="right" colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">
            <table width="100%" border="0" class="b-collapse">
                <!-- Table Row Start -->
                <tr align="right" class="text-grey">
                    <td width="50%" class="subtotal">@lang('modules.invoices.subTotal')</td>
                </tr>
                <!-- Table Row End -->
            @if ($discount != 0 && $discount != '')
                <!-- Table Row Start -->
                    <tr align="right" class="text-grey">
                        <td width="50%" class="subtotal">@lang('modules.invoices.discount')
                        </td>
                    </tr>
                    <!-- Table Row End -->
            @endif
            @foreach ($taxes as $key => $tax)
                <!-- Table Row Start -->
                    <tr align="right" class="text-grey">
                        <td width="50%" class="subtotal">{{ mb_strtoupper($key) }}</td>
                    </tr>
                    <!-- Table Row End -->
            @endforeach
            <!-- Table Row Start -->
                <tr align="right" class="text-grey">
                    <td width="50%" class="total">@lang('modules.invoices.total')</td>
                </tr>
                <!-- Table Row End -->
            @if ($invoice->amountDue())
                <!-- Table Row Start -->
                    <tr align="right" class="balance text-black">
                        <td width="50%" class="balance-left">@lang('modules.invoices.total')
                            @lang('modules.invoices.due')</td>
                    </tr>
                    <!-- Table Row End -->
                @endif

            </table>
        </td>
        <td class="total-box" align="right"
            width="{{ $invoiceSetting->hsn_sac_code_show ? '17%' : '20%' }}">
            <table width="100%" class="b-collapse">
                <!-- Table Row Start -->
                <tr align="right" class="text-grey">
                    <td class="subtotal-amt">
                        {{ currency_format($invoice->sub_total, $invoice->currency_id, false) }}</td>
                </tr>
                <!-- Table Row End -->
            @if ($discount != 0 && $discount != '')
                <!-- Table Row Start -->
                    <tr align="right" class="text-grey">
                        <td class="subtotal-amt">
                            {{ currency_format($discount, $invoice->currency_id, false) }}</td>
                    </tr>
                    <!-- Table Row End -->
            @endif
            @foreach ($taxes as $key => $tax)
                <!-- Table Row Start -->
                    <tr align="right" class="text-grey">
                        <td class="subtotal-amt">{{ currency_format($tax, $invoice->currency_id, false) }}
                        </td>
                    </tr>
                    <!-- Table Row End -->
            @endforeach
            <!-- Table Row Start -->
                <tr align="right" class="text-grey">
                    <td class="total-amt f-15">
                        {{ currency_format($invoice->total, $invoice->currency_id, false) }}</td>
                </tr>
                <!-- Table Row End -->
            @if ($invoice->amountDue())
                <!-- Table Row Start -->
                    <tr align="right" class="balance text-black">
                        <td class="balance-right">
                            {{ currency_format($invoice->amountDue(), $invoice->currency_id, false) }}
                            {{ $invoice->currency->currency_code }}</td>
                    </tr>
                    <!-- Table Row End -->
                @endif
            </table>
        </td>
    </tr>
</table>

<table class="bg-white" border="0" cellpadding="0" cellspacing="0" width="100%" role="presentation">
    <tbody>
    <tr class="text-grey">
        @if ($invoiceSetting->authorised_signatory && $invoiceSetting->authorised_signatory_signature && $invoice->status == 'paid')
            <td class="" align="right">
                <img style="height:95px; margin-bottom: -50px; margin-top: 5px;"
                     src="{{ $invoiceSetting->authorised_signatory_signature_url }}"
                     alt="{{ mb_ucwords($company->company_name) }}"
                     id="logo"/><br>
                @lang('modules.invoiceSettings.authorisedSignatory')
            </td>
        @endif

    </tr>
    <!-- Table Row Start -->
    @if (!($invoiceSetting->authorised_signatory && $invoiceSetting->authorised_signatory_signature && $invoice->status == 'paid'))
        <tr>
            <td height="10"></td>
        </tr>
    @endif
    <tr>
        <td class="f-11">
            @lang('modules.invoiceSettings.invoiceTerms')</td>
    </tr>
    <!-- Table Row End -->
    <!-- Table Row Start -->
    <tr class="text-grey">
        <td class="f-11 line-height">{!! nl2br($invoiceSetting->invoice_terms) !!}</td>
    </tr>
    @if ($invoice->note != '')
        <tr>
            <td height="10"></td>
        </tr>
        <tr>
            <td class="f-11">@lang('app.note')</td>
        </tr>
        <!-- Table Row End -->
        <!-- Table Row Start -->
        <tr class="text-grey">
            <td class="f-11 line-height word-break">{!! $invoice->note ? nl2br($invoice->note) : '--' !!}</td>
        </tr>
    @endif
    <!-- Table Row End -->
    @if (isset($taxes) && $invoiceSetting->tax_calculation_msg == 1)
        <!-- Table Row Start -->
        <tr class="text-grey">
            <td width="100%" class="f-11 line-height">
                <p class="text-dark-grey">
                    @if ($invoice->calculate_tax == 'after_discount')
                        @lang('messages.calculateTaxAfterDiscount')
                    @else
                        @lang('messages.calculateTaxBeforeDiscount')
                    @endif
                </p>
            </td>
        </tr>
        <!-- Table Row End -->
    @endif
    <!-- Table Row End -->
    </tbody>
</table>

{{--Custom fields data--}}
@if(isset($fields) && count($fields) > 0)
    <div class="page_break"></div>
    <h3 class="box-title m-t-20 text-center h3-border"> @lang('modules.projects.otherInfo')</h3>
    <table class="bg-white" border="0" cellspacing="0" cellpadding="0" width="100%" role="presentation">
        @foreach($fields as $field)
            <tr>
                <td style="text-align: left;background: none;">
                    <div class="f-14">{{ ucfirst($field->label) }}</div>
                    <p class="f-14 line-height text-grey">
                        @if( $field->type == 'text' || $field->type == 'password' || $field->type == 'number' || $field->type == 'textarea')
                            {{$invoice->custom_fields_data['field_'.$field->id] ?? '-'}}
                        @elseif($field->type == 'radio')
                            {{ !is_null($invoice->custom_fields_data['field_'.$field->id]) ? $invoice->custom_fields_data['field_'.$field->id] : '-' }}
                        @elseif($field->type == 'select')
                            {{ (!is_null($invoice->custom_fields_data['field_'.$field->id]) && $invoice->custom_fields_data['field_'.$field->id] != '') ? $field->values[$invoice->custom_fields_data['field_'.$field->id]] : '-' }}
                        @elseif($field->type == 'checkbox')
                            {{ !is_null($invoice->custom_fields_data['field_'.$field->id]) ? $invoice->custom_fields_data['field_'.$field->id] : '-' }}
                        @elseif($field->type == 'date')
                            {{ !is_null($invoice->custom_fields_data['field_'.$field->id]) ? \Carbon\Carbon::parse($invoice->custom_fields_data['field_'.$field->id])->translatedFormat($invoice->company->date_format) : '--'}}
                        @endif
                    </p>
                </td>
            </tr>
        @endforeach
    </table>
    </div>

@endif

@if (count($payments) > 0)
    <div class="page_break"></div>
    <h3>@lang('app.menu.payments') ({{ $invoice->invoice_number }})</h3>
    <table class="f-14 b-collapse" width="100%">
        <tr class="main-table-heading text-grey">
            <td class="text-center">#</td>
            <td class="text-center">@lang('modules.invoices.price')</td>
            <td class="text-center">@lang('modules.invoices.paymentMethod')</td>
            <td class="text-center">@lang('modules.invoices.paidOn')</td>
        </tr>

        @forelse($payments as $key => $payment)
            <tr class="main-table-items">
                <td class="text-center">{{ $key + 1 }}</td>
                <td class="text-center">{{ currency_format($payment->amount, $invoice->currency_id, false) }}
                    {{ $invoice->currency->currency_code }}</td>
                <td class="text-center">
                    @php
                        $method = '--';

                        if (!is_null($payment->offline_method_id)) {
                            $method = $payment->offlineMethod->name;
                        } elseif (isset($payment->gateway)) {
                            $method = $payment->gateway;
                        }
                    @endphp

                    {{ $method }}
                </td>
                <td class="text-center"> {{ $payment->paid_on->translatedFormat($company->date_format) }} </td>
            </tr>
        @empty
            <tr>
                <td colspan="4">@lang('messages.noRecordFound') </td>
            </tr>
        @endforelse
    </table>
@endif

</body>

</html>
