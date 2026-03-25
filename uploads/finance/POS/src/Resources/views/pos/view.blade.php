@extends('layouts.main')
@section('page-title')
    {{__('POS Order Detail')}}
@endsection
@push('scripts')
    <script>
        $(document).on('click', '#shipping', function () {
            var url = $(this).data('url');
            var is_display = $("#shipping").is(":checked");
            $.ajax({
                url: url,
                type: 'get',
                data: {
                    'is_display': is_display,
                },
                success: function (data) {
                }
            });
        })
    </script>
@endpush
@push('css')
<link rel="stylesheet" href="{{ asset('packages/workdo/Pos/src/Resources/assets/css/custom.css') }}" id="main-style-link">
@endpush
@section('page-breadcrumb')
{{__('POS Order')}},
{{\workdo\Pos\Entities\Pos::posNumberFormat($pos->pos_id) }}
@endsection
@section('page-action')
    @permission('pos order manage')
        <div class="float-end">
            <a href="{{ route('pos.pdf', Crypt::encrypt($pos->id))}}" class="btn btn-sm btn-primary" target="_blank" data-bs-toggle="tooltip" data-title="{{ __('Download') }}" title="{{ __('Download') }}"><i class="ti ti-download"></i></a>
        </div>
    @endpermission
@endsection
@section('content')
    <div class="card mt-3">
        <div class="card-body">
            <div class="invoice">
                <div class="invoice-print">
                    <div class="d-flex flex-wrap align-items-center justify-content-between row-gap invoice-title border-1 border-bottom  pb-3 mb-3">
                        <div>
                            <h2 class="h3 mb-0">{{ __('POS') }}</h2>
                        </div>
                        <div>
                            <div class="d-flex invoice-wrp flex-wrap align-items-center gap-md-2 gap-1">
                                <div class="d-flex invoice-date flex-wrap align-items-center gap-md-3 gap-1">
                                    <p class="mb-0"> <strong>{{__('Issue Date')}} :</strong>
                                        {{company_date_formate($pos->pos_date)}}</p>
                                </div>
                                <h3 class="invoice-number mb-0">
                                    {{ \workdo\Pos\Entities\Pos::posNumberFormat($pos->pos_id) }}
                                </h3>
                            </div>
                        </div>
                    </div>
                    @if(!empty($customer->billing_name) &&  !empty($customer->shipping_name))
                    <div class="p-sm-4 p-3 invoice-billed">
                        <div class="row row-gap">
                            <div class="col-lg-4 col-sm-6">
                                @if(!empty($customer->billing_name))
                                    <p class="mb-2">
                                        <strong class="h5 mb-1 d-block">{{__('Billed To')}} :</strong>
                                        <span class="text-muted d-block" style="max-width:80%">
                                            {{ !empty($customer->billing_name) ? $customer->billing_name : '' }}
                                            {{ !empty($customer->billing_address) ? $customer->billing_address : '' }}
                                            {{ !empty($customer->billing_city) ? $customer->billing_city . ' ,' : '' }}
                                            {{ !empty($customer->billing_state) ? $customer->billing_state . ' ,' : '' }}
                                            {{ !empty($customer->billing_zip) ? $customer->billing_zip : '' }}
                                            {{ !empty($customer->billing_country) ? $customer->billing_country : '' }}

                                        </span>
                                    </p>
                                    <p class="mb-1 text-dark">
                                        {{ !empty($customer->billing_phone) ? $customer->billing_phone : '' }}
                                    </p>
                                    <p class="mb-0">
                                        <strong>{{__('Tax Number ')}} : </strong>{{!empty($customer->tax_number)?$customer->tax_number:''}}
                                    </p>
                                @endif
                            </div>
                            <div class="col-lg-4 col-sm-6">
                                @if(isset($company_setting['pos_shipping_display']) && $company_setting['pos_shipping_display']=='on')
                                    @if(!empty($customer->shipping_name))
                                        <p class="mb-2">
                                            <strong class="h5 mb-1 d-block">{{__('Shipped To')}} :</strong>
                                            <span class="text-muted d-block" style="max-width:80%">
                                                {{ !empty($customer->shipping_name) ? $customer->shipping_name : '' }}
                                                {{ !empty($customer->shipping_address) ? $customer->shipping_address : '' }}
                                                {{ !empty($customer->shipping_city) ? $customer->shipping_city .' ,': '' }}
                                                {{ !empty($customer->shipping_state) ? $customer->shipping_state .' ,': '' }}
                                                {{ !empty($customer->shipping_zip) ? $customer->shipping_zip : '' }}
                                                {{ !empty($customer->shipping_country) ? $customer->shipping_country : '' }}

                                            </span>
                                        </p>
                                            <p class="mb-1 text-dark">
                                                {{ !empty($customer->shipping_phone) ? $customer->shipping_phone : '' }}
                                            </p>
                                            <p class="mb-0">
                                                <strong>{{__('Tax Number ')}} : </strong>{{!empty($customer->tax_number)?$customer->tax_number:''}}
                                            </p>

                                    @endif
                                @endif
                            </div>
                        </div>
                    </div>
                    @endif
                    <div class="invoice-summary  mt-3">
                        <div class="invoice-title border-1 border-bottom mb-3 pb-2">
                            <h3 class="h4 mb-0">{{ __('Item Summary') }}</h3>
                        </div>
                        <div class="table-responsive mt-2">
                            <table class="table mb-0 table-striped">
                                <thead>
                                    <tr>
                                        <th class="text-white bg-primary text-uppercase" >#</th>
                                        <th class="text-white bg-primary text-uppercase">{{__('Items')}}</th>
                                        <th class="text-white bg-primary text-uppercase">{{__('Quantity')}}</th>
                                        <th class="text-white bg-primary text-uppercase">{{__('Price')}}</th>
                                        <th class="text-white bg-primary text-uppercase">{{__('Tax')}}</th>
                                        <th class="text-white bg-primary text-uppercase">{{__('Tax Amount')}}</th>
                                        <th class="text-white bg-primary text-uppercase">{{__('Total')}}</th>
                                    </tr>
                                </thead>
                                @php
                                    $totalQuantity=0;
                                    $totalRate=0;
                                    $totalTaxPrice=0;
                                    $totalDiscount=0;
                                    $taxesData=[];
                                @endphp
                                @foreach($iteams as $key =>$iteam)
                                    @if(!empty($iteam->tax))
                                    @php
                                                    $taxes=\workdo\Pos\Entities\Pos::tax($iteam->tax);
                                                    $totalQuantity+=$iteam->quantity;
                                                    $totalRate+=$iteam->price;
                                                    $totalDiscount+=$iteam->discount;
                                                    foreach($taxes as $taxe){
                                                        $taxDataPrice=\workdo\Pos\Entities\Pos::taxRate($taxe->rate,$iteam->price,$iteam->quantity);
                                                        if (array_key_exists($taxe->name,$taxesData))
                                                        {
                                                            $taxesData[$taxe->name] = $taxesData[$taxe->name]+$taxDataPrice;
                                                        }
                                                        else
                                                        {
                                                            $taxesData[$taxe->name] = $taxDataPrice;
                                                        }
                                                    }
                                                @endphp
                                            @endif
                                    <tr>
                                        <td>{{$key+1}}</td>
                                        <td>{{!empty($iteam->product)?$iteam->product->name:''}}</td>
                                        <td>{{$iteam->quantity}}</td>
                                        <td>{{currency_format_with_sym($iteam->price)}}</td>
                                        <td>
                                            @if(!empty($iteam->tax))
                                                <table class="w-100">
                                                    <tbody>
                                                    @php
                                                        $totalTaxRate = 0;
                                                        $totalTaxPrice = 0;
                                                    @endphp
                                                    @foreach($taxes as $tax)
                                                        @php
                                                            $taxPrice=\workdo\Pos\Entities\Pos::taxRate($tax->rate,$iteam->price,$iteam->quantity);
                                                            $totalTaxPrice+=$taxPrice;
                                                        @endphp
                                                        <tr>

                                                            <td>{{$tax->name .' ('.$tax->rate .'%)'}}</td>
                                                        </tr>
                                                    @endforeach
                                                    </tbody>
                                                </table>
                                            @else
                                                -
                                            @endif
                                        </td>
                                        <td>{{currency_format_with_sym($totalTaxPrice)}}</td>
                                        <td >{{currency_format_with_sym(($iteam->price*$iteam->quantity+$totalTaxPrice))}}</td>
                                    </tr>
                                @endforeach

                                <tfoot>
                                    <tr>
                                        <td colspan="5"></td>
                                        <td><b>{{__(' Sub Total')}}</b></td>
                                        <td>{{currency_format_with_sym($posPayment['amount'])}}</td>
                                    </tr>
                                    <tr>
                                        <td colspan="5"></td>
                                        <td><b>{{__('Discount')}}</b></td>
                                        <td>{{currency_format_with_sym($posPayment['discount'])}}</td>
                                    </tr>
                                    <tr>
                                        <td colspan="5"></td>
                                        <td><b>{{__('Total')}}</b></td>
                                        <td>{{currency_format_with_sym($posPayment['discount_amount'])}}</td>
                                    </tr>
                                </tfoot>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection
