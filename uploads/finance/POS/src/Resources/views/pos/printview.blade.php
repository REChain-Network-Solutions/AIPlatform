@php
    $company_settings = getCompanyAllSetting();
@endphp
<div class="pt-0 pb-3 modal-body pos-module" id="printarea">
    <table class="table pos-module-tbl">
        <tbody>
        <div class="text-center ">
            <h3>{{isset($company_settings['company_name']) ? $company_settings['company_name'] : ''}}</h3>
        </div>
        <br>

        <div class="text-left">
            <b>{{ $details['pos_id'] }}</b>
        </div>
        <div class="text-left">
            {{isset($company_settings['company_name']) ? $company_settings['company_name'] : ''}}<br>
            {{isset($company_settings['company_email']) ? $company_settings['company_email'] : ''}}<br>
            {{isset($company_settings['company_address']) ? $company_settings['company_address'] : ''}}<br>
            {{isset($company_settings['company_city']) ? $company_settings['company_city'] : ''}},
            {{isset($company_settings['company_state']) ? $company_settings['company_state'] : ''}},
            {{isset($company_settings['company_zipcode']) ? $company_settings['company_zipcode'] : ''}}<br>
            {{isset($company_settings['company_country']) ? $company_settings['company_country'] : ''}}<br>
            {{isset($company_settings['company_telephone']) ? $company_settings['company_telephone'] : ''}}<br>
        </div>
        <div class="invoice-to mt-2 product-border" >
            {!! isset($details['customer']['name']) ? '' : $details['customer']['details'] !!}
        </div><br>
        <div>
            {!! isset($details['customer']['name']) ? 'Name:  ' . $details['customer']['name'] : '' !!}
        </div>
        <div>
            {!! isset($details['customer']['address']) ? 'Address:  ' . $details['customer']['address'] : '' !!}
        </div>
        <div>
            {!! isset($details['customer']['email']) ? 'Email:  ' . $details['customer']['email'] : '' !!}
        </div>
        <div>
            {!! isset($details['customer']['phone_number']) ? 'Phone:  ' . $details['customer']['phone_number'] : '' !!}
        </div>
        <div>
            {!! isset($details['date']) ? 'Date of POS:  ' . $details['date'] : '' !!}
        </div>
        <div class="product-border">
            {!! isset($details['warehouse']['details']) ? 'Warehouse Name:  ' . $details['warehouse']['details'] : '' !!}
        </div>
        </tbody>
    </table>
    <div class=" text-black text-left fs-5 mt-0 mb-0">{{__('Items')}}</div>
    @if (array_key_exists('data', $sales))
        @foreach ($sales['data'] as $key => $value)
            <div class="mt-2">
                <div class="p-0"> <b>{{ $value['name'] }}</b></div>
                <div class="d-flex product-border">
                    <div>{{ __('Quantity:') }}</div>
                    <div class="text-end ms-auto">{{ $value['quantity'] }}</div>
                </div>
            </div>
            <div class="d-flex product-border">
                <div>{{__('Price:')}}</div>
                <div class="text-end ms-auto">{{ $value['price'] }}</div>
            </div>
            <div class="d-flex product-border">
                <div>{{__('Tax:')}}</div>
                <div class="text-end ms-auto"> {{ $value['tax'] }}</div>
            </div>
            <div class="d-flex product-border mb-2">
                <div>{{__('Tax Amount:')}}</div>
                <div class="text-end ms-auto">{{ $value['tax_amount'] }}</div>
            </div>
            <div class="d-flex product-border mb-2">
                <div>{{__('Sub Total:')}}</div>
                <div class="text-end ms-auto"> {{ $value['subtotal'] }}</div>
            </div>
        @endforeach
    @endif
    <div class="d-flex product-border mb-2 mt-4">
        <div><b>{{__('Discount:')}}</b></div>
        <div class="text-end ms-auto"> {{ $sales['discount'] }}</div>
    </div>
    <div class="d-flex product-border mb-2">
        <div><b>{{__('Total:')}}</b></div>
        <div class="text-end ms-auto"> {{ $sales['total'] }}</div>
    </div>

    <h5 class="text-center mt-3 font-label">{{__('Thank You For Shopping With Us. Please visit again.')}}</h5>
</div>

<div class="justify-content-center pt-2 modal-footer">
    <a href="#" id="print"
       class="btn btn-primary btn-sm text-right float-right mb-3 ">
        {{ __('Print') }}
    </a>
</div>
<script>
    $("#print").click(function () {
        var print_div = document.getElementById("printarea");
        $('.row').addClass('d-none');
        $('.toast').addClass('d-none');
        $('#print').addClass('d-none');
        window.print();
        $('.row').removeClass('d-none');
        $('#print').removeClass('d-none');
        $('.toast').removeClass('d-none');
    });
</script>




