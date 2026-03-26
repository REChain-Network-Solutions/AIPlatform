<html lang="en">

<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>@lang('app.invoice')</title>
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
            $font = 'THSarabun';
        } else if($invoiceSetting->is_chinese_lang) {
            $font = 'SimHei';
        }else {
            $font = 'noto-sans';
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

        .clearfix:after {
            content: "";
            display: table;
            clear: both;
        }

        a {
            color: #0087C3;
            text-decoration: none;
        }

        body {
            position: relative;
            width: 100%;
            height: auto;
            margin: 0 auto;
            color: #555555;
            background: #FFFFFF;
            font-size: 14px;
            font-family: Verdana, Arial, Helvetica, sans-serif;
        }

        h2 {
            font-weight: normal;
        }

        header {
            padding: 10px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid #AAAAAA;
        }

        #logo {
            float: left;
            margin-top: 11px;
        }

        #logo img {
            height: 55px;
            margin-bottom: 15px;
        }

        #company {}

        #details {
            margin-bottom: 50px;
        }

        #client {
            padding-left: 6px;
            float: left;
        }

        #client .to {
            color: #777777;
        }

        h2.name {
            font-size: 1.2em;
            font-weight: normal;
            margin: 0;
        }

        #invoice {}

        #invoice h1 {
            color: #0087C3;
            font-size: 2.4em;
            line-height: 1em;
            font-weight: normal;
            margin: 0 0 10px 0;
        }

        #invoice .date {
            font-size: 1.1em;
            color: #777777;
        }

        table {
            width: 100%;
            border-spacing: 0;
            margin-bottom: 20px;
        }

        table th,
        table td {
            padding: 5px 10px 7px 10px;
            background: #EEEEEE;
            text-align: center;
            border-bottom: 1px solid #FFFFFF;
        }

        table th {
            white-space: nowrap;
            font-weight: normal;
        }

        table td {
            text-align: right;
        }

        table td.desc h3,
        table td.qty h3 {
            color: #57B223;
            font-size: 1.2em;
            font-weight: normal;
            margin: 0 0 0 0;
        }

        table .no {
            color: #FFFFFF;
            font-size: 1.6em;
            background: #57B223;
            width: 10%;
        }

        table .desc {
            text-align: left;
        }

        table .unit {
            background: #DDDDDD;
        }


        table .total {
            background: #57B223;
            color: #FFFFFF;
        }

        table td.unit,
        table td.qty,
        table td.total {
            font-size: 1.2em;
            text-align: center;
        }

        table td.unit {
            width: 35%;
        }

        table td.desc {
            width: 45%;
        }

        table td.qty {
            width: 5%;
        }

        .status {
            margin-top: 15px;
            padding: 1px 8px 5px;
            font-size: 1.3em;
            width: 80px;
            color: #fff;
            float: right;
            text-align: center;
            display: inline-block;
        }

        .status.unpaid {
            background-color: #E7505A;
        }

        .status.paid {
            background-color: #26C281;
        }

        .status.cancelled {
            background-color: #95A5A6;
        }

        .status.error {
            background-color: #F4D03F;
        }

        table tr.tax .desc {
            text-align: right;
            color: #1BA39C;
        }

        table tr.discount .desc {
            text-align: right;
            color: #E43A45;
        }

        table tr.subtotal .desc {
            text-align: right;
            color: #1d0707;
        }

        table tbody tr:last-child td {
            border: none;
        }

        table tfoot td {
            padding: 10px 10px 20px 10px;
            background: #FFFFFF;
            border-bottom: none;
            font-size: 1.2em;
            white-space: nowrap;
            border-bottom: 1px solid #AAAAAA;
        }

        table tfoot tr:first-child td {
            border-top: none;
        }

        table tfoot tr td:first-child {
            border: none;
        }

        #thanks {
            font-size: 2em;
            margin-bottom: 50px;
        }

        #notices {
            padding-left: 6px;
            border-left: 6px solid #0087C3;
        }

        #notices .notice {
            font-size: 1.2em;
        }

        footer {
            color: #777777;
            width: 100%;
            height: 30px;
            position: absolute;
            bottom: 0;
            border-top: 1px solid #AAAAAA;
            padding: 8px 0;
            text-align: center;
        }

        table.billing td {
            background-color: #fff;
        }

        table td div#invoiced_to {
            text-align: left;
        }

        #notes {
            color: #767676;
            font-size: 11px;
        }

        .item-summary {
            font-size: 12px
        }

        .mb-3 {
            margin-bottom: 1rem;
        }

        .logo {
            text-align: right;
        }

        .logo img {
            max-width: 150px !important;
        }

    </style>
</head>

<body>
    <header class="clearfix">

        <table cellpadding="0" cellspacing="0" class="billing">
            <tr>
                <td>
                    <div id="invoiced_to">
                        @if ($invoice->project && $invoice->project->client && $invoice->project->client->clientDetails && ($invoice->project->client->name || $invoice->project->client->email || $invoice->project->client->mobile || $invoice->project->client->clientDetails->company_name || $invoice->project->client->clientDetails->address) && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))

                            <small>@lang('modules.invoices.billedTo'):</small>

                            @if ($invoice->project->client->name && $invoiceSetting->show_client_name == 'yes')
                                <h3 class="name">{{ mb_ucwords($invoice->project->client->name) }}</h3>
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
                                <h3 class="name">
                                    {{ mb_ucwords($invoice->project->client->clientDetails->company_name) }}</h3>
                            @endif

                            @if ($invoice->project->client->clientDetails->address && $invoiceSetting->show_client_company_address == 'yes')
                                <div class="mb-3">
                                    <b>@lang('app.address')</b>
                                    <div>{!! nl2br($invoice->project->clientDetails->address) !!}</div>
                                </div>
                            @endif

                            @if ($invoice->show_shipping_address === 'yes')
                                <div>
                                    <b>@lang('app.shippingAddress') :</b>
                                    <div>{!! nl2br($invoice->project->clientDetails->shipping_address) !!}</div>
                                </div>
                            @endif

                            @if ($invoiceSetting->show_gst == 'yes' && !is_null($invoice->project->client->clientDetails->gst_number))
                                <div> @lang('app.gstIn'): {{ $invoice->project->client->clientDetails->gst_number }}
                                </div>
                            @endif
                        @elseif($invoice->client && $invoice->clientDetails && ($invoice->client->name || $invoice->client->email || $invoice->client->mobile || $invoice->clientDetails->company_name || $invoice->clientDetails->address) && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))
                            <small>@lang('modules.invoices.billedTo'):</small>

                            @if ($invoice->client->name && $invoiceSetting->show_client_name == 'yes')
                                <h3 class="name">{{ mb_ucwords($invoice->client->name) }}</h3>
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
                                <h3 class="name">{{ mb_ucwords($invoice->clientDetails->company_name) }}</h3>
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
                                <div> @lang('app.gstIn'): {{ $invoice->clientDetails->gst_number }} </div>
                            @endif
                        @endif

                        @if (is_null($invoice->project) && $invoice->estimate && $invoice->estimate->client && $invoice->estimate->client->clientDetails && ($invoiceSetting->show_client_name == 'yes' || $invoiceSetting->show_client_email == 'yes' || $invoiceSetting->show_client_phone == 'yes' || $invoiceSetting->show_client_company_name == 'yes' || $invoiceSetting->show_client_company_address == 'yes'))
                            <small>@lang('modules.invoices.billedTo'):</small>
                            @if ($invoice->estimate->client->name && $invoiceSetting->show_client_name == 'yes')
                                <h3 class="name">{{ mb_ucwords($invoice->estimate->client->name) }}</h3>
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
                                <h3 class="name">
                                    {{ mb_ucwords($invoice->estimate->client->clientDetails->company_name) }}</h3>
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
                                <div> @lang('app.gstIn'): {{ $invoice->estimate->client->clientDetails->gst_number }}
                                </div>
                            @endif
                        @endif

                        @if (($invoiceSetting->show_project == 1) && (isset($invoice->project->project_name)))
                                <small>@lang('modules.invoices.projectName')</small><br>
                                {{ $invoice->project->project_name }}
                        @endif
                    </div>
                </td>
                <td>
                    <div id="company">
                        <div class="logo">
                            @if ($invoiceSetting->logo)
                                <img src="{{ asset_url('app-logo/' . $invoiceSetting->logo) }}" alt="home"
                                    class="dark-logo" />
                            @else
                                <img src="{{ asset_url('app-logo/' . $company->logo) }}" alt="home"
                                    class="dark-logo" />
                            @endif
                        </div>
                        <small>@lang('modules.invoices.generatedBy'):</small>
                        <h3 class="name">{{ mb_ucwords($company->company_name) }}</h3>
                        @if (!is_null($company))
                            <div>{!! nl2br($defaultAddress->address) !!}</div>
                            <div>{{ $company->company_phone }}</div>
                        @endif
                        @if ($invoiceSetting->show_gst == 'yes' && $invoice->address)
                            <div>{{ $invoice->address->tax_name }}: {{ $invoice->address->tax_number }}</div>
                        @endif
                    </div>
                </td>
            </tr>
        </table>
    </header>
    <main>
        <div id="details" class="clearfix">

            <div id="invoice">
                <h1>{{ $invoice->invoice_number }}</h1>
                @if ($creditNote)
                    <div class="">@lang('app.credit-note'): {{ $creditNote->cn_number }}</div>
                @endif
                <div class="date">@lang('app.menu.issues') @lang('app.date'):
                    {{ $invoice->issue_date->translatedFormat($company->date_format) }}</div>
                @if ($invoice->status === 'unpaid')
                    <div class="date">@lang('app.dueDate'):
                        {{ $invoice->due_date->translatedFormat($company->date_format) }}</div>
                @endif
                <div class="">@lang('app.status'): {{ mb_ucwords($invoice->status) }}</div>
            </div>

        </div>
        <table border="0" cellspacing="0" cellpadding="0">
            <thead>
                <tr>
                    <th class="no">#</th>
                    <th class="desc">@lang('modules.invoices.item')</th>
                    <th class="qty">@lang('modules.invoices.qty')</th>
                    <th class="qty">@lang('modules.invoices.unitPrice') ({!! htmlentities($invoice->currency->currency_code) !!})</th>
                    <th class="unit">@lang('modules.invoices.price') ({!! htmlentities($invoice->currency->currency_code) !!})</th>
                </tr>
            </thead>
            <tbody>
                <?php $count = 0; ?>
                @foreach ($invoice->items as $item)
                    @if ($item->type == 'item')
                        <tr style="page-break-inside: avoid;">
                            <td class="no">{{ ++$count }}</td>
                            <td class="desc">
                                <h3>{{ ucfirst($item->item_name) }}</h3>
                                @if (!is_null($item->item_summary))
                                    <p class="item-summary">{!! nl2br(strip_tags($item->item_summary, ['p', 'b', 'strong', 'a'])) !!}</p>
                                @endif
                            </td>
                            <td class="qty">
                                <h3>{{ $item->quantity }}</h3>
                            </td>
                            <td class="qty">
                                <h3>{{ number_format((float) $item->unit_price, 2, '.', '') }}</h3>
                            </td>
                            <td class="unit">{{ number_format((float) $item->amount, 2, '.', '') }}</td>
                        </tr>
                    @endif
                @endforeach
                <tr style="page-break-inside: avoid;" class="subtotal">
                    <td class="no">&nbsp;</td>
                    <td class="qty">&nbsp;</td>
                    <td class="qty">&nbsp;</td>
                    <td class="desc">@lang('modules.invoices.subTotal')</td>
                    <td class="unit">{{ number_format((float) $invoice->sub_total, 2, '.', '') }}</td>
                </tr>
                @if ($discount != 0 && $discount != '')
                    <tr style="page-break-inside: avoid;" class="discount">
                        <td class="no">&nbsp;</td>
                        <td class="qty">&nbsp;</td>
                        <td class="qty">&nbsp;</td>
                        <td class="desc">@lang('modules.invoices.discount')</td>
                        <td class="unit">{{ number_format((float) $discount, 2, '.', '') }}</td>
                    </tr>
                @endif
                @foreach ($taxes as $key => $tax)
                    <tr style="page-break-inside: avoid;" class="tax">
                        <td class="no">&nbsp;</td>
                        <td class="qty">&nbsp;</td>
                        <td class="qty">&nbsp;</td>
                        <td class="desc">{{ mb_strtoupper($key) }}</td>
                        <td class="unit">{{ number_format((float) $tax, 2, '.', '') }}</td>
                    </tr>
                @endforeach
            </tbody>
            <tfoot>
                <tr dontbreak="true">
                    <td colspan="4">@lang('modules.invoices.total')</td>
                    <td style="text-align: center">{{ number_format((float) $invoice->total, 2, '.', '') }}</td>
                </tr>
                @if ($invoice->appliedCredits() > 0)
                    <tr dontbreak="true">
                        <td colspan="4">@lang('modules.invoices.appliedCredits')</td>
                        <td style="text-align: center">
                            {{ number_format((float) $invoice->appliedCredits(), 2, '.', '') }}</td>
                    </tr>
                @endif
                <tr dontbreak="true">
                    <td colspan="4">@lang('modules.invoices.total') @lang('modules.invoices.paid')</td>
                    <td style="text-align: center">{{ number_format((float) $invoice->getPaidAmount(), 2, '.', '') }}
                    </td>
                </tr>
                <tr dontbreak="true">
                    <td colspan="4">@lang('modules.invoices.total') @lang('modules.invoices.due')</td>
                    <td style="text-align: center">{{ number_format((float) $invoice->amountDue(), 2, '.', '') }}
                    </td>
                </tr>
            </tfoot>
        </table>
        <p>&nbsp;</p>
        <hr>
        <p id="notes">
            @if (!is_null($invoice->note))
                @lang('app.note') <br>{!! nl2br($invoice->note) !!}<br>
            @endif
            @if ($invoice->status == 'unpaid')
                @lang('modules.invoiceSettings.invoiceTerms') <br>{!! nl2br($invoiceSetting->invoice_terms) !!}
            @endif

        </p>

    </main>
</body>

</html>
