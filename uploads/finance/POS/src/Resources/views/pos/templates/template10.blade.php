<!DOCTYPE html>
<html lang="en" dir="{{ $settings['site_rtl'] == 'on'?'rtl':''}}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ \workdo\Pos\Entities\Pos::posNumberFormat($pos->pos_id)}} | {{ !empty(company_setting('title_text')) ? company_setting('title_text') : (!empty(admin_setting('title_text')) ? admin_setting('title_text') :'WorkDo') }}</title>
    <link
        href="https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap"
        rel="stylesheet">


    <style type="text/css">
        html[dir="rtl"]  {
                letter-spacing: 0.1px;
            }
        :root {
            --theme-color: {{$color}};
            --white: #ffffff;
            --black: #000000;
        }

        body {
            font-family: 'Lato', sans-serif;
        }

        p,
        li,
        ul,
        ol {
            margin: 0;
            padding: 0;
            list-style: none;
            line-height: 1.5;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        table tr th {
            padding: 0.75rem;
            text-align: left;
        }

        table tr td {
            padding: 0.75rem;
            text-align: left;
        }

        table th small {
            display: block;
            font-size: 12px;
        }

        .invoice-preview-main {
            max-width: 700px;
            width: 100%;
            margin: 0 auto;
            background: #ffff;
            box-shadow: 0 0 10px #ddd;
        }

        .invoice-logo {
            max-width: 200px;
            width: 100%;
        }

        .invoice-header table td {
            padding: 15px 30px;
        }

        .text-right {
            text-align: right;
        }

        .no-space tr td {
            padding: 0;
          white-space: nowrap;
        }

        .vertical-align-top td {
            vertical-align: top;
        }

        .view-qrcode {
            max-width: 114px;
            height: 114px;
            margin-left: auto;
            margin-top: 15px;
            background: var(--white);
        }

        .view-qrcode img {
            width: 100%;
            height: 100%;
        }

        .invoice-body {
            padding: 30px 25px 0;
        }

        table.add-border tr {
            border-top: 1px solid var(--theme-color);
        }

        tfoot tr:first-of-type {
            border-bottom: 1px solid var(--theme-color);
        }

        .total-table tr:first-of-type td {
            padding-top: 0;
        }

        .total-table tr:first-of-type {
            border-top: 0;
        }

        .sub-total {
            padding-right: 0;
            padding-left: 0;
        }

        .border-0 {
            border: none !important;
        }

        .invoice-summary td,
        .invoice-summary th {
            font-size: 13px;
            font-weight: 600;
        }

        .total-table td:last-of-type {
            width: 146px;
        }

        .invoice-footer {
            padding: 15px 20px;
        }

        .itm-description td {
            padding-top: 0;
        }
        html[dir="rtl"] table tr td,
        html[dir="rtl"] table tr th{
            text-align: right;
        }
        html[dir="rtl"]  .text-right{
            text-align: left;
        }
        html[dir="rtl"] .view-qrcode{
            margin-left: 0;
            margin-right: auto;
        }
        p:not(:last-of-type){
            margin-bottom: 15px;
        }
        .invoice-summary p{
            margin-bottom: 0;
        }
    </style>
</head>

<body>
    <div class="invoice-preview-main" id="boxes">
        <div class="invoice-header">
            <table class="vertical-align-top">
                <tbody>
                    <tr>
                        <td >
                            <h3 style=" display: inline-block; text-transform: uppercase; font-size: 40px; font-weight: bold; border-top: 5px solid var(--theme-color); padding-top: 5px;">{{ __('POS')}}</h3>

                        </td>
                        <td class="text-right">
                            <img class="invoice-logo"
                                 src="{{$img}}"
                                alt="">
                        </td>

                    </tr>
                </tbody>
            </table>
            <table class="vertical-align-top">
                <tbody>
                    <tr>
                        @if (!empty($settings['company_name']) && !empty($settings['company_email']) && !empty($settings['company_address']))
                            <td>
                                <p>
                                <b style="color: var(--theme-color);">{{ __('FROM')}}:</b><br>
                                @if(!empty($settings['company_name'])){{$settings['company_name']}}@endif<br>
                                    @if(!empty($settings['company_email'])){{$settings['company_email']}}@endif<br>
                                    @if(!empty($settings['company_telephone'])){{$settings['company_telephone']}}@endif<br>
                                    @if(!empty($settings['company_address'])){{$settings['company_address']}}@endif
                                    @if(!empty($settings['company_city'])) <br> {{$settings['company_city']}}, @endif
                                    @if(!empty($settings['company_state'])){{$settings['company_state']}}@endif
                                    @if(!empty($settings['company_country'])) <br>{{$settings['company_country']}}@endif
                                    @if(!empty($settings['company_zipcode'])) - {{$settings['company_zipcode']}}@endif<br>
                                    @if(!empty($settings['registration_number'])){{__('Registration Number')}} : {{$settings['registration_number']}} @endif<br>
                                    @if(!empty($settings['tax_type']) && !empty($settings['vat_number'])){{$settings['tax_type'].' '. __('Number')}} : {{$settings['vat_number']}} <br>@endif
                                </p>
                            </td>
                        @endif
                        <td>
                            <table class="no-space"  style="width: 45%;margin-left: auto;">
                                <tbody>
                                    <tr>
                                        <td>{{ __('Number: ')}}</td>
                                        <td class="text-right">{{\workdo\Pos\Entities\Pos::posNumberFormat($pos->pos_id)}}</td>
                                    </tr>
                                    <tr>
                                        <td>{{ __('POS Date:')}}</td>
                                        <td class="text-right">{{company_date_formate($pos->pos_date)}}</td>
                                    </tr>
                                    <tr>
                                        <td colspan="2">

                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="invoice-body">
            <table>
                <tbody>
                    <tr>
                        <td >
                            <strong style="margin-bottom: 10px; display:block;">{{ __('Bill To')}}:</strong>
                            <p>
                                {{ !empty($customer->billing_name) ? $customer->billing_name : '' }}<br>
                                {{ !empty($customer->billing_address) ? $customer->billing_address : '' }}<br>
                                {{ !empty($customer->billing_city) ? $customer->billing_city . ' ,' : '' }}
                                {{ !empty($customer->billing_state) ? $customer->billing_state . ' ,' : '' }}
                                {{ !empty($customer->billing_zip) ? $customer->billing_zip : '' }}<br>
                                {{ !empty($customer->billing_country) ? $customer->billing_country : '' }}<br>
                                {{ !empty($customer->billing_phone) ? $customer->billing_phone : '' }}<br>
                            </p>
                        </td>
                        @if($settings['pos_shipping_display']=='on')
                            <td class="text-right">
                                <strong style="margin-bottom: 10px; display:block;">{{__('Ship To')}}:</strong>
                                <p>
                                    {{ !empty($customer->shipping_name) ? $customer->shipping_name : '' }}<br>
                                    {{ !empty($customer->shipping_address) ? $customer->shipping_address : '' }}<br>
                                    {{ !empty($customer->shipping_city) ? $customer->shipping_city .' ,': '' }}
                                    {{ !empty($customer->shipping_state) ? $customer->shipping_state .' ,': '' }}
                                    {{ !empty($customer->shipping_zip) ? $customer->shipping_zip : '' }}<br>
                                    {{ !empty($customer->shipping_country) ? $customer->shipping_country : '' }}<br>
                                    {{ !empty($customer->shipping_phone) ? $customer->shipping_phone : '' }}<br>
                                </p>
                            </td>
                        @endif
                    </tr>
                </tbody>
            </table>
            <table class="add-border invoice-summary" style="margin-top: 30px;">
                <thead style="background-color: var(--theme-color);color:  {{$font_color}};">
                    <tr>
                        <th>{{__('Item')}}</th>
                        <th>{{__('Quantity')}}</th>
                        <th>{{__('Price')}}</th>
                        <th>{{__('Tax')}} (%)</th>
                        <th>{{__('Tax Amount')}}</th>
                        <th>{{__('Total')}}</th>
                    </tr>
                </thead>
                <tbody>
                    @if(isset($pos->itemData) && count($pos->itemData) > 0)
                    @foreach($pos->itemData as $key => $item)
                    <tr>
                        <td>{{$item->name}}</td>
                        <td>{{$item->quantity}}</td>
                        <td>{{currency_format_with_sym($item->price)}}</td>
                        <td>
                            @php
                                $totalTaxRate = 0;
                                $totalTaxPrice=0;
                            @endphp
                            @if(!empty($item->itemTax))
                                @foreach($item->itemTax as $taxes)
                                    @php
                                        $res = str_ireplace( array( '%' ), ' ', $taxes['rate']);
                                        $taxPrice=\workdo\Pos\Entities\Pos::taxRate($res,$item->price,$item->quantity);
                                        $totalTaxPrice+=$taxPrice;
                                    @endphp
                                        <span>{{$taxes['name']}}</span> <span>({{$taxes['rate']}})</span><br>
                                @endforeach
                            @else
                                -
                            @endif
                        </td>
                        <td>{{currency_format_with_sym($totalTaxPrice)}}</td>
                        <td >{{currency_format_with_sym(($item->price*$item->quantity)+$totalTaxPrice)}}</td>
                    </tr>
                @endforeach
                    @else
                        <tr>
                            <td>-</td>
                            <td>-</td>
                            <td>-</td>
                            <td>
                                <p>-</p>
                                <p>-</p>
                            </td>
                            <td>-</td>
                            <td>-</td>
                        <tr class="border-0 itm-description ">
                                <td colspan="6">-</td>
                            </tr>
                        </tr>
                    @endif
                </tbody>
                <tfoot>
                    <tr>
                        <td>{{__('Total')}}</td>
                        <td>{{$pos->totalQuantity}}</td>
                        <td>{{currency_format_with_sym($pos->totalRate)}}</td>
                        <td>-</td>
                        <td>{{currency_format_with_sym($pos->totalTaxPrice) }}</td>
                        <td>{{currency_format_with_sym($posPayment->amount)}}</td>
                    </tr>
                    <tr>
                        <td colspan="4"></td>
                        <td colspan="2" class="sub-total">
                            <table class="total-table">
                                <tr>
                                    <td>{{__('Subtotal')}}:</td>
                                    <td>{{currency_format_with_sym($posPayment->amount)}}</td>
                                </tr>
                                <tr>
                                    <td>{{__('Discount')}}:</td>
                                    @if(!empty($posPayment->discount))
                                        <td>{{currency_format_with_sym($posPayment->discount)}}</td>
                                    @else
                                        <td>0</td>
                                    @endif
                                </tr>
                                <tr>
                                    <td>{{__('Total')}}:</td>
                                    @if(!empty($posPayment->discount))
                                        <td>{{currency_format_with_sym($posPayment->amount - $posPayment->discount)}}</td>
                                    @else
                                       <td>{{currency_format_with_sym($posPayment->amount)}}</td>
                                    @endif
                                </tr>
                            </table>
                        </td>
                    </tr>
                </tfoot>
            </table>
            <div class="invoice-footer">
                <p>
                    {{$settings['pos_footer_title']}} <br>
                    {{$settings['pos_footer_notes']}}
                </p>
            </div>
        </div>
    </div>
    @if(!isset($preview))
     @include('pos::pos.script');
    @endif
</body>

</html>
