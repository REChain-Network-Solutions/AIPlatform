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
        font-family: {{$font}}, Arial, Helvetica, sans-serif;
    }

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
            font: inherit;
            font-size: 12px;
            vertical-align: baseline;
            /* font-family: Verdana, Arial, Helvetica, sans-serif; */
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
            border-spacing: 0;
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

        .x-hidden {
            display: none !important;
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
            margin: 0 auto;
            position: relative;
        }

        .right-invoice {
            padding: 40px 30px;
        }

        #memo .company-info {
            float: left;
        }

        #memo .company-info {
            font-size: 12px;
            min-width: 20px;
            line-height: 15px;
        }

        #memo .company-info span {
            font-size: 12px;
            display: inline-block;
            min-width: 20px;
        }

        #memo .logo {
            float: right;
            margin-left: 15px;
        }

        #memo .logo img {
            height: 50px;
        }

        #memo:after {
            content: '';
            display: block;
            clear: both;
        }

        #invoice-title-number {
            margin: 20px 0 20px 0;
            display: inline-block;
            float: left;
        }

        #invoice-title-number .title-top {
            font-size: 15px;
            margin-bottom: 5px;
        }

        #invoice-title-number .title-top span {
            display: inline-block;
            min-width: 20px;
        }

        #invoice-title-number .title-top #number {
            text-align: right;
            float: right;
            color: #858585;
        }

        #invoice-title-number .title-top:after {
            content: '';
            display: block;
            clear: both;
        }

        #invoice-title-number #title {
            display: inline-block;
            background: #415472;
            color: white;
            font-size: 25px !important;
            padding: 8px 13px;
        }

        #client-info {
            text-align: right;
            min-width: 220px;
            line-height: 15px;
            font-size: 12px;
        }

        #client-info>div {
            margin-bottom: 3px;
            min-width: 20px;
        }

        #client-info span {
            display: block;
            min-width: 20px;
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

        #invoice-info {
            float: left;
            margin-top: 10px;
            line-height: 18px;
        }

        #invoice-info div {
            margin-bottom: 3px;
        }

        #invoice-info div span {
            display: inline-block;
            min-width: 20px;
            min-height: 18px;
        }

        #invoice-info div span:first-child {
            font-weight: bold;
            margin-right: 10px;
        }

        .currency {
            margin-top: 20px;
            text-align: right;
            color: #858585;
            font-style: italic;
            font-size: 12px;
        }

        .currency span {
            display: inline-block;
            min-width: 20px;
        }

        #items {
            margin-top: 10px;
        }

        #items .first-cell,
        #items table th:first-child,
        #items table td:first-child {
            width: 18px;
            text-align: right;
        }

        #items table {
            border-collapse: separate;
            width: 100%;
        }

        #items table th {
            font-size: 12px;
            padding: 5px 3px;
            text-align: center;
            background: #b0b4b3;
            color: white;
        }

        #items table th:nth-child(2) {
            width: 30%;
            text-align: left;
        }

        #items table th:last-child {
            /*text-align: right;*/
        }

        #items table td {
            padding: 10px 3px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }

        #items table td:first-child {
            text-align: left;
        }

        #items table td:nth-child(2) {
            text-align: left;
        }

        #sums {
            margin: 25px 30px 0 0;
            width: 100%;
        }

        #sums table {
            width: 70%;
            float: right;
        }

        #sums table tr th,
        #sums table tr td {
            min-width: 100px;
            padding: 9px 8px;
            text-align: right;
        }

        #sums table tr th {
            width: 70%;
            font-weight: bold;
        }

        #sums table tr td.last {
            min-width: 0 !important;
            max-width: 0 !important;
            width: 0 !important;
            padding: 0 !important;
            border: none !important;
        }

        #sums table tr.amount-total td,
        #sums table tr.amount-total th {
            font-size: 20px !important;
        }

        #sums table tr.due-amount th,
        #sums table tr.due-amount td {
            font-weight: bold;
        }

        #sums:after {
            content: '';
            display: block;
            clear: both;
        }

        #terms {
            margin-top: 20px !important;
            font-size: 12px;
        }

        .calculate_tax {
            margin-top: 20px !important;
            font-size: 12px;
        }

        #terms>span {
            font-weight: bold;
            display: inline-block;
            min-width: 20px;
        }

        #terms>div {
            min-height: 50px;
            min-width: 50px;
        }

        #terms .notes {
            min-height: 30px;
            min-width: 50px;
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

        #itemsPayment {
            margin-top: 10px;
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
            font-size: 12px;
            padding: 5px 3px;
            text-align: center;
            background: #b0b4b3;
            color: white;
        }

        #itemsPayment table th:nth-child(2) {
            width: 30%;
            text-align: left;
        }

        #itemsPayment table th:last-child {
            /*text-align: right;*/
        }

        #itemsPayment table td {
            padding: 10px 3px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }

        #itemsPayment table td:first-child {
            text-align: left;
        }

        #itemsPayment table td:nth-child(2) {
            text-align: left;
        }

        #itemsPayment,
        .box-title {
            margin: 25px 30px 0 30px;
        }

        .word-break {
            word-wrap: break-word;
        }

        .left-stripes {
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            width: 100px;
            background: url("{{ asset("img/stripe-bg.jpg") }}") repeat;
        }
        .left-stripes .circle {
            -moz-border-radius: 50%;
            -webkit-border-radius: 50%;
            border-radius: 50%;
            background: #415472;
            width: 30px;
            height: 30px;
            position: absolute;
            left: 33%;
        }
        .left-stripes .circle.c-upper {
            top: 440px;
        }
        .left-stripes .circle.c-lower {
            top: 690px;
        }

        #notes {
            color: #767676;
            font-size: 11px;
            margin-top: 10px;
            margin-left: 40px;
        }

        #field-title {
            color: #504f4f;
            font-size: 13px;
            margin-top: 10px;
            margin-left: 40px;
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

            #memo .client-logo {
            float:left;
            display: flex !important;
            flex-direction: column !important;
            position: absolute;
            /* margin-bottom: 15px; */
            }

            #memo .client-logo img {
                height: 50px;
                margin-bottom: 10px;
            }

        @endif

    </style>

</head>

<body>
    <div id="container">

        <div class="right-invoice">
            <section id="memo">
                <div class="client-logo">
                    <div>
                        <img style="height:50px;" src="{{ $invoiceSetting->logo_url }}" />
                    </div>
                    <div class="company-info description">
                            <br>
                        <div class="description">
                            {{ mb_ucwords($company->company_name) }}
                        </div>
                        @if ($company->company_email)
                        <span class="description">{{ $company->company_email }}</span>
                            <br>
                        @endif
                        @if ($company->company_phone)
                        <span class="description">{{ $company->company_phone }}</span>
                            <br>
                        @endif
                        @if ($invoice->address)
                            <span class="description">{!! nl2br($invoice->address->address) !!}</span>
                        @endif
                    </div>
                </div>

                <div class="logo" id="client-info" >
                    @if($invoice->clientDetails->company_logo)
                        <div>
                            <img src="{{ $invoice->clientDetails->image_url }}"
                                alt="{{ mb_ucwords($invoice->clientDetails->company_name) }}" />
                        </div>
                    @endif

                    @if ($invoice->project && $invoice->project->client && $invoice->project->client->clientDetails && ($invoice->project->client->name || $invoice->project->client->email || $invoice->project->client->mobile || $invoice->project->client->clientDetails->company_name || $invoice->project->client->clientDetails->address) && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))
                        <section class="description">
                            <div class="description">@lang('modules.invoices.billedTo'):</div>
                            @if ($invoice->project->client->name && $invoiceSetting->show_client_name == 'yes')
                                <div class="client-name">
                                    <strong>{{ mb_ucwords($invoice->project->client->name) }}</strong>
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
                                    <span>{!! nl2br($invoice->project->clientDetails->address) !!}</span>
                                </div>
                            @endif

                            @if ($invoice->show_shipping_address === 'yes')
                                <div>
                                    <span>
                                        <b>@lang('app.shippingAddress') :</b><br>
                                        {!! nl2br($invoice->project->client->clientDetails->shipping_address) !!}
                                    </span>
                                </div>
                            @endif

                            <div>
                                <span>{{ $invoice->project->client->email }}</span>
                            </div>
                            @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->project->client->clientDetails) && !is_null($invoice->project->client->clientDetails->gst_number))
                                <div>
                                    <span> @lang('app.gstIn'): {{ $invoice->project->client->clientDetails->gst_number }}
                                    </span>
                                </div>
                            @endif
                        </section>
                    @elseif(!is_null($invoice->client_id))
                        <section id="client-info" class="description">
                            <span class="description">@lang('modules.invoices.billedTo'): </span>
                            <div class="client-name">
                                {{ mb_ucwords($invoice->client->name) }}
                            </div>

                            @if ($invoice->clientDetails)
                                <div>
                                    <span>{{ mb_ucwords($invoice->clientDetails->company_name) }}</span>
                                </div>
                                <div class="mb-3">
                                    <span>
                                        @lang('app.address') :
                                        {!! nl2br($invoice->clientDetails->address) !!}
                                    </span>
                                </div>
                                @if ($invoice->show_shipping_address === 'yes')
                                    <div>
                                        <span>
                                            <b>@lang('app.shippingAddress') :</b><br>
                                            {!! nl2br($invoice->clientDetails->shipping_address) !!}
                                        </span>
                                    </div>
                                @endif
                            @endif

                            <div>
                                <span>{{ $invoice->client->email }}</span>
                            </div>
                            @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->clientDetails) && !is_null($invoice->clientDetails->gst_number))
                                <div>
                                    <span> @lang('app.gstIn'): {{ $invoice->clientDetails->gst_number }} </span>
                                </div>
                            @endif
                        </section>
                    @endif



                    @if (is_null($invoice->project) && !is_null($invoice->estimate) && !is_null($invoice->estimate->client))
                        <section id="client-info">
                            <span>@lang('modules.invoices.billedTo'):</span>
                            <div class="client-name">
                                <strong>{{ mb_ucwords($invoice->estimate->client->name) }}</strong>
                            </div>

                            <div>
                                <span>{{ mb_ucwords($invoice->estimate->client->clientDetails->company_name) }}</span>
                            </div>
                            <div class="mb-3">
                                <span>
                                    <b>@lang('app.address') :</b><br>
                                    {!! nl2br($invoice->estimate->client->clientDetails->address) !!}
                                </span>
                            </div>
                            @if ($invoice->show_shipping_address === 'yes')
                                <div>
                                    <span>
                                        <b>@lang('app.shippingAddress') :</b><br>
                                        {!! nl2br($invoice->estimate->client->clientDetails->shipping_address) !!}
                                    </span>
                                </div>
                            @endif
                            <div>
                                <span>{{ $invoice->estimate->client->email }}</span>
                            </div>
                            @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->estimate->client->clientDetails->gst_number))
                                <div>
                                    <span> @lang('app.gstIn'): {{ $invoice->estimate->client->clientDetails->gst_number }}
                                    </span>
                                </div>
                            @endif
                        </section>
                    @endif

                </div>
            </section>

            <section id="invoice-title-number">

                <div class="title-top description" >
                    <span>@lang('modules.invoices.invoiceDate'):</span>
                    <span>{{ $invoice->issue_date->translatedFormat($company->date_format) }}</span>
                </div>

                <div id="title">{{ $invoice->invoice_number }}</div>

            </section>
      


            <div class="clearfix"></div>

            <section id="invoice-info">
                @if (empty($invoice->order_id) && $invoice->status === 'unpaid' && $invoice->due_date->year > 1)
                    <div>
                        <span>@lang('app.dueDate'):</span>
                        <span>{{ $invoice->due_date->translatedFormat($company->date_format) }}</span>
                    </div>
                @endif
                @if ($invoiceSetting->show_status)
                <div>
                    <span>@lang('app.status'):</span> <span>{{ mb_ucwords($invoice->status) }}</span>
                </div>
                @endif

                @if ($creditNote)
                    <div>
                        <span>@lang('app.credit-note'):</span> <span>{{ $creditNote->cn_number }}</span>
                    </div>
                @endif

                @if ($invoiceSetting->show_project == 1 && isset($invoice->project->project_name))
                <div>
                    <span>@lang('modules.invoices.projectName'):</span> <span> {{ $invoice->project->project_name }} </span>
                </div>
                @endif

            </section>

            <div class="clearfix"></div>
            <p>&nbsp;</p>
            <p>&nbsp;</p>
            <p>&nbsp;</p>

            <section id="items">

                <table cellpadding="0" cellspacing="0" >

                    <tr>
                        <th>#</th> <!-- Dummy cell for the row number and row commands -->
                        <th class="description">@lang('modules.invoices.item')</th>
                        @if ($invoiceSetting->hsn_sac_code_show)
                            <th>@lang('app.hsnSac')</th>
                        @endif
                        <th class="description">{{ $invoice->unit->unit_type }}</th>
                        <th class="description">@lang('modules.invoices.unitPrice')</th>
                        <th class="description">@lang('modules.invoices.tax')</th>
                        <th class="description">@lang('modules.invoices.price') ({!! htmlentities($invoice->currency->currency_code) !!})</th>
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
                                        <p class="item-summary mb-3">{!! nl2br(strip_tags($item->item_summary, ['p', 'b', 'strong', 'a'])) !!}</p>
                                    @endif
                                    @if ($item->invoiceItemImage)
                                        <p>
                                            <img src="{{ $item->invoiceItemImage->file_url }}" width="80" height="80"
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

            </section>

            <section id="sums">

                <table cellpadding="0" cellspacing="0">
                    <tr>
                        <th>@lang('modules.invoices.subTotal'):</th>
                        <td>{{ currency_format($invoice->sub_total, $invoice->currency_id, false) }}</td>
                    </tr>
                    @if ($discount != 0 && $discount != '')
                        <tr data-iterate="tax">
                            <th>@lang('modules.invoices.discount'):</th>
                            <td>{{ currency_format($discount, $invoice->currency_id, false) }}</td>
                        </tr>
                    @endif
                    @foreach ($taxes as $key => $tax)
                        <tr data-iterate="tax">
                            <th>{{ mb_strtoupper($key) }}:</th>
                            <td>{{ currency_format($tax, $invoice->currency_id, false) }}</td>
                        </tr>
                    @endforeach
                    <tr class="amount-total">
                        <th>@lang('modules.invoices.total'):</th>
                        <td>{{ currency_format($invoice->total, $invoice->currency_id, false) }}</td>
                    </tr>
                    @if ($invoice->creditNotes()->count() > 0)
                        <tr>
                            <th>
                                @lang('modules.invoices.appliedCredits'):</th>
                            <td>
                                {{ currency_format($invoice->appliedCredits(), $invoice->currency_id, false) }}
                            </td>
                        </tr>
                    @endif
                    <tr>
                        <th>@lang('modules.invoices.total') @lang('modules.invoices.paid'):</th>
                        <td> {{ currency_format($invoice->getPaidAmount(), $invoice->currency_id, false) }}</td>
                    </tr>
                    @if ($invoice->amountDue())
                    <tr>
                        <th>@lang('modules.invoices.total') @lang('modules.invoices.due'):</th>
                        <td> {{ currency_format($invoice->amountDue(), $invoice->currency_id, false) }}</td>
                    </tr>
                    @endif
                    @if ($invoiceSetting->authorised_signatory && $invoiceSetting->authorised_signatory_signature && $invoice->status == 'paid')
                    <tr>
                        <td colspan="2" style="font-size:15px" align="right">
                            <img style="height:95px; margin-bottom: -50px; margin-top: 5px;"
                            src="{{ $invoiceSetting->authorised_signatory_signature_url }}" alt="{{ mb_ucwords($company->company_name) }}"/><br>
                            @lang('modules.invoiceSettings.authorisedSignatory')
                        </td>
                    </tr>
                    @endif
                </table>

            </section>


            <div class="clearfix"></div>

            <section id="terms">

                <div class="notes word-break description">
                    @if (!is_null($invoice->note))
                        <br>@lang('app.note') <br> {!! '--' !!}
                    @endif
                    @if ($invoice->status == 'unpaid')
                        <br><br>@lang('modules.invoiceSettings.invoiceTerms') <br>{!! nl2br($invoiceSetting->invoice_terms) !!}
                    @endif
                </div>

            </section>
            @if (isset($taxes) && $invoiceSetting->tax_calculation_msg == 1)
                <div class="clearfix"></div>
                <section class="calculate_tax" >
                    <div class="description">
                        @if ($invoice->calculate_tax == 'after_discount')
                            @lang('messages.calculateTaxAfterDiscount')
                        @else
                            @lang('messages.calculateTaxBeforeDiscount')
                        @endif
                    </div>
                </section>
            @endif
        </div>
    </div>

    {{--Custom fields data--}}
    @if(isset($fields) && count($fields) > 0)
        <div class="page_break"></div>
        <div id="container">
            <div class="invoice-body right-invoice b-all m-b-20">
                <h3 class="box-title m-t-20 text-center h3-border"> @lang('modules.projects.otherInfo')</h3>
                <table  style="background: none" border="0" cellspacing="0" cellpadding="0" width="100%">
                    @foreach($fields as $field)
                        <tr>
                            <td style="text-align: left;background: none;" >
                                <div id="field-title">{{ ucfirst($field->label) }}</div>
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
            </div>

        </div>

    @endif

    @if (count($payments) > 0)
        <div class="page_break"></div>
        <div id="container">
            <div class="invoice-body right-invoice b-all m-b-20">
                <h3 class="box-title m-t-20 text-center h3-border description">@lang('app.menu.payments')</h3>
                <div class="row">
                    <div class="col-sm-12">
                        <div class="table-responsive m-t-40" id="itemsPayment" style="clear: both;">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th class="text-center">#</th>
                                        <th class="text-center description">@lang('modules.payments.amount')</th>
                                        <th class="text-center description">@lang('modules.invoices.paymentMethod')</th>
                                        <th class="text-center description">@lang('modules.invoices.paidOn')</th>
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
                                            <td class="text-center description">
                                                {{ $payment->paid_on->translatedFormat($company->date_format) }} </td>
                                        </tr>
                                    @empty
                                    @endforelse
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    @endif

</body>

</html>
