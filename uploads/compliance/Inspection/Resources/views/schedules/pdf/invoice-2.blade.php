<!DOCTYPE html>
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>

    <title>@lang('app.invoice')</title>

    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <meta name="description" content="Invoice">

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
            body
        {
            font-weight: normal !important;
        }
        @endif
        * {
            font-family: {{$font}}, DejaVu Sans , sans-serif;
        }

        /* Reset styles */
        html,
        body,
        div,
        span,
        applet,
        object,
        iframe,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        p,
        blockquote,
        pre,
        a,
        abbr,
        acronym,
        address,
        big,
        cite,
        code,
        del,
        dfn,
        em,
        img,
        ins,
        kbd,
        q,
        s,
        samp,
        small,
        strike,
        strong,
        sub,
        sup,
        tt,
        var,
        b,
        u,
        i,
        center,
        dl,
        dt,
        dd,
        ol,
        ul,
        li,
        fieldset,
        form,
        label,
        legend,
        table,
        caption,
        tbody,
        tfoot,
        thead,
        tr,
        th,
        td,
        article,
        aside,
        canvas,
        details,
        embed,
        figure,
        figcaption,
        footer,
        header,
        hgroup,
        menu,
        nav,
        output,
        ruby,
        section,
        summary,
        time,
        mark,
        audio,
        video {
            margin: 0;
            padding: 0;
            border: 0;
            /* font-family: Verdana, Arial, Helvetica, sans-serif; */
            /* font-size: 80%; */
            vertical-align: baseline;
        }

        html {
            line-height: 1;
        }

        ol,
        ul {
            list-style: none;
        }

        table {
            border-collapse: collapse;
        }

        caption,
        th,
        td {
            text-align: left;
            font-weight: normal;
            vertical-align: middle;
        }

        q,
        blockquote {
            quotes: none;
        }

        q:before,
        q:after,
        blockquote:before,
        blockquote:after {
            content: "";
            content: none;
        }

        a img {
            border: none;
        }

        article,
        aside,
        details,
        figcaption,
        figure,
        footer,
        header,
        hgroup,
        main,
        menu,
        nav,
        section,
        summary {
            display: block;
        }

        /* Invoice styles */
        /**
         * DON'T override any styles for the <html> and <body> tags, as this may break the layout.
         * Instead wrap everything in one main <div id="container"> element where you may change
         * something like the font or the background of the invoice
         */
        html,
        body {
            /* MOVE ALONG, NOTHING TO CHANGE HERE! */
        }

        /**
         * IMPORTANT NOTICE: DON'T USE '!important' otherwise this may lead to broken print layout.
         * Some browsers may require '!important' in oder to work properly but be careful with it.
         */
        .clearfix {
            display: block;
            clear: both;
        }

        .hidden {
            display: none;
        }

        b,
        strong,
        .bold {
            font-weight: bold;
        }

        #container {
            font: normal 13px/1.4em 'Open Sans', Sans-serif;
            margin: 0 auto;
        }

        .invoice-top {
            color: #000000;
            padding: 40px 40px 10px 40px;
        }

        .invoice-body {
            padding: 10px 40px 40px 40px;
        }

        #memo .logo {
            float: left;
            margin-right: 20px;
        }

        #memo .logo img {
            height: 50px;
        }

        #memo .company-info {
            /*float: right;*/
            text-align: right;
        }

        #memo .company-info .company-name {
            font-size: 20px;
            text-align: right;
        }

        #memo .company-info .spacer {
            height: 15px;
            display: block;
        }

        #memo .company-info div {
            font-size: 12px;
            text-align: right;
            line-height: 18px;
        }

        #memo:after {
            content: '';
            display: block;
            clear: both;
        }

        #invoice-info {
            text-align: left;
            margin-top: 20px;
            line-height: 18px;
        }

        #invoice-info table {
            width: 30%;
        }

        #invoice-info>div {
            float: left;
        }

        #invoice-info>div>span {
            display: block;
            min-width: 100px;
            min-height: 18px;
            margin-bottom: 3px;
        }

        #invoice-info:after {
            content: '';
            display: block;
            clear: both;
        }

        #client-info {
            text-align: right;
            min-width: 220px;
            line-height: 18px;
        }

        #client-info>div {
            margin-bottom: 3px;
        }

        #client-info span {
            display: block;
        }

        #client-info>span {
            margin-bottom: 3px;
        }

        #invoice-title-number {
            margin-top: 30px;
        }

        #invoice-title-number #title {
            font-size: 35px;
        }

        #invoice-title-number #number {
            text-align: left;
            font-size: 20px;
        }

        table {
            table-layout: fixed;
        }

        table th,
        table td {
            vertical-align: top;
            word-break: keep-all;
            word-wrap: break-word;
        }

        #items .first-cell,
        #items table th:first-child,
        #items table td:first-child {
            width: 18px;
            text-align: right;
        }

        #items table {
            border-collapse: collapse;
            width: 100%;
            border: 1px solid #000000
        }

        #items table th {
            font-weight: bold;
            padding: 12px 10px;
            text-align: right;
            border-bottom: 1px solid #444;
        }

        #items table th:nth-child(2) {
            width: 30%;
            text-align: left;
        }

        #items table th:last-child {
            text-align: right;
        }

        #items table td {
            border-right: 1px solid #b6b6b6;
            padding: 7px 10px;
            text-align: right;
        }

        #items table td:first-child {
            text-align: left;
        }

        #items table td:nth-child(2) {
            text-align: left;
        }

        #items table td:last-child {
            border-right: none !important;
        }

        #terms>div {
            min-height: 30px;
        }

        .payment-info {
            color: #707070;
            font-size: 12px;
        }

        .payment-info div {
            display: inline-block;
            min-width: 10px;
        }

        .ib_drop_zone {
            color: #F8ED31 !important;
            border-color: #F8ED31 !important;
        }

        .item-summary {
            font-size: 11px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .mb-3 {
            margin-bottom: 1rem;
        }

        /**
         * If the printed invoice is not looking as expected you may tune up
         * the print styles (you can use !important to override styles)
         */
        @media print {
            /* Here goes your print styles */
        }

        .page_break {
            page-break-before: always;
        }

        .h3-border {
            border-bottom: 1px solid #AAAAAA;
        }

        table td.text-center {
            text-align: center;
        }

        table td.text-right {
            text-align: right;
        }

        #itemsPayment .first-cell,
        #itemsPayment table th:first-child,
        #itemsPayment table td:first-child {
            width: 18px;
            text-align: right;
        }

        #itemsPayment table {
            border-collapse: separate;
            width: 100%;
        }

        #itemsPayment table th {
            font-weight: bold;
            padding: 12px 10px;
            text-align: right;
            border-bottom: 1px solid #444;
            text-transform: uppercase;
        }

        #itemsPayment table th:nth-child(2) {
            width: 30%;
            text-align: left;
        }

        #itemsPayment table th:last-child {
            text-align: right;
        }

        #itemsPayment table td {
            border-right: 1px solid #b6b6b6;
            padding: 15px 10px;
            text-align: right;
        }

        #itemsPayment table td:first-child {
            text-align: left;
            /*border-right: none !important;*/
        }

        #itemsPayment table td:nth-child(2) {
            text-align: left;
        }

        #itemsPayment table td:last-child {
            border-right: none !important;
        }

        .word-break {
            word-wrap: break-word;
        }

        #notes {
            color: #767676;
            font-size: 11px;
        }

        @if($invoiceSetting->locale == 'th')

            table td {
            font-weight: bold !important;
            font-size: 20px !important;
            }

            .description
            {
                font-weight: bold !important;
                font-size: 16px !important;
            }
        @endif

        .client-logo {
            height: 80px;
        }

        .client-logo-div {
            position: absolute;
            margin-left: -80px;
            margin-top: 30px;

        }


    </style>
</head>

<body>
    <div id="container" class="description">
        <div class="invoice-top">
            <section id="memo"  class="description">
                <div class="logo">
                    <img src="{{ $invoiceSetting->logo_url }}" />
                </div>

                <div class="company-info description">
                    <span class="company-name">
                        {{ mb_ucwords($company->company_name) }}
                    </span>

                    <span class="spacer"></span>
                    @if ($company->company_email)
                    <div>{{ $company->company_email }}</div>
                    @endif
                    @if ($company->company_phone)
                    <div>{{ $company->company_phone }}</div>
                    @endif
                    @if ($invoice->address)
                        <div>{!! nl2br($invoice->address->address) !!}</div>
                    @endif

                    <span class="clearfix"></span>



                    <span class="clearfix"></span>

                    @if ($invoiceSetting->show_gst == 'yes' && $invoice->address)
                        <div>{{ $invoice->address->tax_name }}: {{ $invoice->address->tax_number }}</div>
                    @endif


                </div>

            </section>

            <section id="invoice-info"  class="description">
                <table>
                    <tr>
                        <td>@lang('modules.invoices.invoiceDate'):</td>
                        <td>{{ $invoice->issue_date->translatedFormat($company->date_format) }}</td>
                    </tr>
                    @if (empty($invoice->order_id) && $invoice->status === 'unpaid' && $invoice->due_date->year > 1)
                        <tr>
                            <td>@lang('app.dueDate'):</td>
                            <td>{{ $invoice->due_date->translatedFormat($company->date_format) }}</td>
                        </tr>
                    @endif
                    @if ($invoiceSetting->show_status)
                    <tr>
                        <td>@lang('app.status'):</td>
                        <td>{{ mb_ucwords($invoice->status) }}</td>
                    </tr>
                    @endif
                    @if ($creditNote)
                        <tr>
                            <td>@lang('app.credit-note')</td>
                            <td>{{ $creditNote->cn_number }}</td>
                        </tr>
                    @endif

                </table>

                <section id="invoice-title-number">

                    <span id="number">{{ $invoice->invoice_number }}</span>

                    @if($invoice->clientDetails->company_logo)
                        <div class="client-logo-div">
                            <img src="{{ $invoice->clientDetails->image_url }}"
                                alt="{{ mb_ucwords($invoice->clientDetails->company_name) }}" class="client-logo"/>
                        </div>
                    @endif


                </section>
            </section>
            @if ($invoice->project && $invoice->project->client && $invoice->project->client->clientDetails && ($invoice->project->client->name || $invoice->project->client->email || $invoice->project->client->mobile || $invoice->project->client->clientDetails->company_name || $invoice->project->client->clientDetails->address) && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))
                <section id="client-info"  class="description">
                    <span>@lang('modules.invoices.billedTo')</span>
                    @if ($invoice->project->client->name && $invoiceSetting->show_client_name == 'yes')
                        <div>
                            <span class="bold">{{ mb_ucwords($invoice->project->client->name) }}</span>
                        </div>
                    @endif

                    @if ($invoice->project->client->email && $invoiceSetting->show_client_email == 'yes')
                        <div>
                            <span class="">{{ $invoice->project->client->email }}</span>
                        </div>
                    @endif

                    @if ($invoice->project->client->mobile && $invoiceSetting->show_client_phone == 'yes')
                        <div>
                            <span class="">{{ $invoice->project->client->mobile }}</span>
                        </div>
                    @endif
                    @if ($invoice->project->client->clientDetails->company_name && $invoiceSetting->show_client_company_name == 'yes')
                        <div>
                            <span>{{ mb_ucwords($invoice->project->client->clientDetails->company_name) }}</span>
                        </div>
                    @endif

                    @if ($invoice->project->client->clientDetails->address && $invoiceSetting->show_client_company_address == 'yes')
                        <div class="mb-3">
                            <b>@lang('app.address')</b>
                            <div>{!! nl2br($invoice->project->clientDetails->address) !!}</div>
                        </div>
                    @endif

                    @if ($invoice->show_shipping_address === 'yes')
                        <div>
                            <b>@lang('app.shippingAddress')</b>
                            <div>{!! nl2br($invoice->project->clientDetails->shipping_address) !!}</div>
                        </div>
                    @endif

                    @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->project->client->clientDetails->gst_number))
                        <div>
                            <span> @lang('app.gstIn'): {{ $invoice->project->client->clientDetails->gst_number }}
                            </span>
                        </div>
                    @endif

                    @if ($invoiceSetting->show_project == 1 && isset($invoice->project->project_name))
                        <div>
                            <span>@lang('modules.invoices.projectName')</span> <span>{{ $invoice->project->project_name }}</span>
                        </div>
                    @endif

                </section>
            @elseif($invoice->client && $invoice->clientDetails && ($invoice->client->name || $invoice->client->email || $invoice->client->mobile || $invoice->clientDetails->company_name || $invoice->clientDetails->address) && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))
                <section id="client-info"  class="description">
                    <span>@lang('modules.invoices.billedTo'):</span>

                    @if ($invoice->client->name && $invoiceSetting->show_client_name == 'yes')
                        <div>
                            <span class="bold">{{ mb_ucwords($invoice->client->name) }}</span>
                        </div>
                    @endif

                    @if ($invoice->client->email && $invoiceSetting->show_client_email == 'yes')
                        <div>
                            <span class="">{{ $invoice->client->email }}</span>
                        </div>
                    @endif

                    @if ($invoice->client->mobile && $invoiceSetting->show_client_phone == 'yes')
                        <div>
                            <span class="">{{ $invoice->client->mobile }}</span>
                        </div>
                    @endif

                    @if ($invoice->clientDetails->company_name && $invoiceSetting->show_client_company_name == 'yes')
                        <div>
                            <span>{{ mb_ucwords($invoice->clientDetails->company_name) }}</span>
                        </div>
                    @endif

                    @if ($invoice->clientDetails->address && $invoiceSetting->show_client_company_address == 'yes')
                        <div class="mb-3">
                            <b>@lang('app.address') :</b>
                            <div>{!! nl2br($invoice->clientDetails->address) !!}</div>
                        </div>
                    @endif

                    @if ($invoice->show_shipping_address === 'yes')
                        <div>
                            <b>@lang('app.shippingAddress') :</b>
                            <div>{!! nl2br($invoice->clientDetails->shipping_address) !!}</div>
                        </div>
                    @endif

                    @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->clientDetails->gst_number))
                        <div>
                            <span> @lang('app.gstIn'): {{ $invoice->clientDetails->gst_number }} </span>
                        </div>
                    @endif
                </section>
            @endif
            @if (is_null($invoice->project) && $invoice->estimate && $invoice->estimate->client && $invoice->estimate->client->clientDetails && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))
                <section id="client-info"  class="description">
                    <span>@lang('modules.invoices.billedTo'):</span>
                    @if ($invoice->estimate->client->name && $invoiceSetting->show_client_name == 'yes')
                        <div>
                            <span class="bold">{{ mb_ucwords($invoice->estimate->client->name) }}</span>
                        </div>
                    @endif

                    @if ($invoice->estimate->client->email && $invoiceSetting->show_client_email == 'yes')
                        <div>
                            <span class="">{{ $invoice->estimate->client->email }}</span>
                        </div>
                    @endif

                    @if ($invoice->estimate->client->mobile && $invoiceSetting->show_client_phone == 'yes')
                        <div>
                            <span class="">{{ $invoice->estimate->client->mobile }}</span>
                        </div>
                    @endif

                    @if ($invoice->estimate->client->clientDetails->company_name && $invoiceSetting->show_client_company_name == 'yes')
                        <div>
                            <span
                                class="">{{ mb_ucwords($invoice->estimate->client->clientDetails->company_name) }}</span>
                        </div>
                    @endif

                    @if ($invoice->estimate->client->clientDetails->address && $invoiceSetting->show_client_company_address == 'yes')
                        <div class="mb-3">
                            <b>@lang('app.address') :</b>
                            <div>{!! nl2br($invoice->estimate->client->clientDetails->address) !!}</div>
                        </div>
                    @endif

                    @if ($invoice->show_shipping_address === 'yes')
                        <div>
                            <b>@lang('app.shippingAddress') :</b>
                            <div>{!! nl2br($invoice->estimate->client->clientDetails->shipping_address) !!}</div>
                        </div>
                    @endif

                    @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->estimate->client->clientDetails->gst_number))
                        <div>
                            <span> @lang('app.gstIn'): {{ $invoice->estimate->client->clientDetails->gst_number }}
                            </span>
                        </div>
                    @endif
                </section>
            @endif


            <div class="clearfix"></div>
        </div>


        <div class="invoice-body">
            <section id="items"  class="description">

                <table cellpadding="0" cellspacing="0">

                    <tr>
                        <th>#</th> <!-- Dummy cell for the row number and row commands -->
                        <th  class="description">@lang('modules.invoices.item')</th>
                        @if ($invoiceSetting->hsn_sac_code_show)
                            <th  class="description">@lang('app.hsnSac')</th>
                        @endif
                        <th  class="description">{{ isset($invoice->unit) ? $invoice->unit->unit_type : 'Qty\hrs' }}</th>
                        <th  class="description">@lang('modules.invoices.unitPrice')</th>
                        <th  class="description">@lang('modules.invoices.tax')</th>
                        <th  class="description">@lang('modules.invoices.price') ({!! htmlentities($invoice->currency->currency_code) !!})</th>
                    </tr>

                    <?php $count = 0; ?>
                    @foreach ($invoice->items as $item)
                        @if ($item->type == 'item')
                            <tr data-iterate="item">
                                <td>{{ ++$count }}</td>
                                <!-- Don't remove this column as it's needed for the row commands -->
                                <td>
                                    {{ ucfirst($item->item_name) }}
                                    @if (!is_null($item->item_summary))
                                        <p class="item-summary">{!! nl2br(strip_tags($item->item_summary, ['p', 'b', 'strong', 'a'])) !!}</p>
                                    @endif
                                    @if ($item->invoiceItemImage)
                                        <p class="mt-2">
                                            <img src="{{ $item->invoiceItemImage->file_url }}" width="60" height="60"
                                                class="img-thumbnail">
                                        </p>
                                    @endif
                                </td>
                                @if ($invoiceSetting->hsn_sac_code_show)
                                    <td>{{ $item->hsn_sac_code ? $item->hsn_sac_code : '--' }}</td>
                                @endif
                                <td>{{ $item->quantity }}</td>
                                <td>{{ currency_format($item->unit_price, $invoice->currency_id, false) }}</td>
                                <td>{{ strtoupper($item->tax_list) }}</td>
                                <td>{{ currency_format($item->amount, $invoice->currency_id, false) }}</td>
                            </tr>
                        @endif
                    @endforeach

                </table>

                <table cellpadding="0" cellspacing="0">
                    <tr>
                        <td colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">@lang('modules.invoices.subTotal'):</td>
                        <td>{{ currency_format($invoice->sub_total, $invoice->currency_id, false) }}</td>
                    </tr>
                    @if ($discount != 0 && $discount != '')
                        <tr data-iterate="tax">
                            <td colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">@lang('modules.invoices.discount'):
                            </td>
                            <td>-{{ currency_format($discount, $invoice->currency_id, false) }}</td>
                        </tr>
                    @endif
                    @foreach ($taxes as $key => $tax)
                        <tr data-iterate="tax">
                            <td colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">
                                {{ mb_strtoupper($key) }}:</td>
                            <td>{{ currency_format($tax, $invoice->currency_id, false) }}</td>
                        </tr>
                    @endforeach
                    <tr class="amount-total">
                        <td colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">
                            @lang('modules.invoices.total'):
                        </td>
                        <td>
                            {{ currency_format($invoice->total, $invoice->currency_id, false) }}
                        </td>
                    </tr>
                    @if ($invoice->creditNotes()->count() > 0)
                        <tr colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">
                            <td>
                                @lang('modules.invoices.appliedCredits'):</td>
                            <td>
                                {{ currency_format($invoice->appliedCredits(), $invoice->currency_id, false) }}
                            </td>
                        </tr>
                    @endif
                    <tr>
                        <td colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">
                            @lang('modules.invoices.total') @lang('modules.invoices.paid'):</td>
                        <td class="text-right">
                            {{ currency_format($invoice->getPaidAmount(), $invoice->currency_id, false) }}
                        </td>
                    </tr>
                    @if ($invoice->amountDue())
                    <tr>
                        <td colspan="{{ $invoiceSetting->hsn_sac_code_show ? '5' : '4' }}">
                            @lang('modules.invoices.total') @lang('modules.invoices.due'):</td>
                        <td class="text-right">
                            {{ currency_format($invoice->amountDue(), $invoice->currency_id, false) }}
                            {{ $invoice->currency->currency_code }}</td>
                    </tr>
                    @endif
                </table>

            </section>
            @if ($invoiceSetting->authorised_signatory && $invoiceSetting->authorised_signatory_signature && $invoice->status == 'paid')
            <section id="terms">
                            <table border="0" cellspacing="0" cellpadding="0" width="100%" style="">
                                <tr>
                                    <td style="font-size:15px; text-align: right">
                                        <img style="height:95px; margin-bottom: -50px; margin-top: 5px;"
                                        src="{{ $invoiceSetting->authorised_signatory_signature_url }}" alt="{{ mb_ucwords($company->company_name) }}"/><br>
                                        @lang('modules.invoiceSettings.authorisedSignatory')
                                    </td>
                                </tr>
                            </table>
                </section>
                @endif
            <section id="terms">
                @if (!is_null($invoice->note))
                    <div class="word-break item-summary description">@lang('app.note') <br>{!! nl2br($invoice->note) !!}</div>
                @endif
                <div class="word-break item-summary description">@lang('modules.invoiceSettings.invoiceTerms') <br>{!! nl2br($invoiceSetting->invoice_terms) !!}</div>

            </section>

            @if (isset($taxes) && $invoiceSetting->tax_calculation_msg == 1)
                <p class="text-dark-grey description">
                    @if ($invoice->calculate_tax == 'after_discount')
                        @lang('messages.calculateTaxAfterDiscount')
                    @else
                        @lang('messages.calculateTaxBeforeDiscount')
                    @endif
                </p>
            @endif

            {{--Custom fields data--}}
            @if(isset($fields) && count($fields) > 0)
                <div class="page_break"></div>
                <h3 class="box-title m-t-20 text-center h3-border"> @lang('modules.projects.otherInfo')</h3>
                <table  style="background: none" border="0" cellspacing="0" cellpadding="0" width="100%">
                    @foreach($fields as $field)
                        <tr>
                            <td style="text-align: left;background: none;" >
                                <div style="font-size: 13px; margin-top: 5px;">{{ ucfirst($field->label) }}</div>
                                <p id="notes">
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
            @endif

            @if (count($payments) > 0)
                <div class="page_break"></div>
                <div class="invoice-body b-all m-b-20 description">
                    <h3 class="box-title m-t-20 text-center h3-border">@lang('app.menu.payments')</h3>
                    <div class="row">
                        <div class="col-sm-12">
                            <div class="table-responsive m-t-40 description" id="itemsPayment" style="clear: both;">
                                <table class="table table-hover">
                                    <thead>
                                        <tr>
                                            <th class="text-center">#</th>
                                            <th class="text-center">@lang('modules.payments.amount')</th>
                                            <th class="text-center">@lang('modules.invoices.paymentMethod')</th>
                                            <th class="text-center">@lang('modules.invoices.paidOn')</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php $count = 0; ?>
                                        @forelse($payments as $key => $payment)
                                            <tr>
                                                <td class="text-center">{{ $key + 1 }}</td>
                                                <td class="text-center">
                                                    {{ number_format((float) $payment->amount, 2, '.', '') }}
                                                    {!! $invoice->currency->currency_code !!} </td>
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
                                                <td class="text-center">
                                                    {{ $payment->paid_on->translatedFormat($company->date_format) }}
                                                </td>
                                            </tr>
                                        @empty
                                        @endforelse
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            @endif

        </div>

    </div>

</body>

</html>
