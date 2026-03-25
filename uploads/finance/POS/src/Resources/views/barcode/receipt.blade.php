@php
    $admin_settings = getAdminAllSetting();
    $company_settings = getCompanyAllSetting();
    $favicon = isset($company_settings['favicon']) ? $company_settings['favicon'] : (isset($admin_settings['favicon']) ? $admin_settings['favicon'] : 'uploads/logo/favicon.png');
@endphp
<!DOCTYPE html>
<html lang="en" dir="{{$company_settings['site_rtl'] == 'on'?'rtl':''}}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">

    <!-- Favicon icon -->
    <link rel="icon" href="{{ check_file($favicon) ? get_file($favicon) : get_file('uploads/logo/favicon.png')  }}{{'?'.time()}}" type="image/x-icon" />

    <link rel="stylesheet" href="{{ asset('assets/css/plugins/main.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/css/plugins/style.css') }}">

    <!-- font css -->
    <link rel="stylesheet" href="{{ asset('assets/fonts/tabler-icons.min.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/fonts/feather.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/fonts/fontawesome.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/fonts/material.css') }}">

    <!-- vendor css -->
    <link rel="stylesheet" href="{{ asset('assets/css/style.css') }}" id="main-style-link">

    <title>{{__('POS Barcode')}} | {{ !empty($company_settings['title_text']) ? $company_settings['title_text'] : (!empty($admin_settings['title_text']) ? $admin_settings['title_text'] :'WorkDo') }}</title>
    @if (isset($company_settings['site_rtl'] ) && $company_settings['site_rtl'] == 'on')
        <link rel="stylesheet" href="{{ asset('assets/css/style-rtl.css')}}" id="main-style-link">
    @endif
</head>
<body>
<div id="bot" class="mt-5">
    <div class="row">
        @foreach($productServices as $product)
            @for($i=1;$i<=$quantity;$i++)
                <div class="col-auto mb-2">
                    <small class="">{{$product->name}}</small>
                    <div data-id="{{$product->id}}" class="product_barcode product_barcode_hight_de product_barcode_{{$product->id}} mt-2" data-skucode="{{ $product->sku }}"></div>
                </div>
            @endfor
        @endforeach
    </div>
</div>
<script>
    setTimeout(() => {
        window.print();
        window.onafterprint = back;

        function back() {
            window.close();
            window.history.back();
        }
    }, 300);
</script>
<script src="{{ asset('js/jquery.min.js') }}"></script>
<script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/jquery-barcode.min.js') }}"></script>
<script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/jquery-barcode.js') }}"></script>
<script>
    $(document).ready(function() {
        $(".product_barcode").each(function() {
            var id = $(this).data("id");
            var sku = $(this).data('skucode');
            sku = encodeURIComponent(sku);
            generateBarcode(sku, id);
        });
    });
    function generateBarcode(val, id) {
        var value = val;
        var btype = '{{ $barcode['barcodeType'] }}';
        var renderer = '{{ $barcode['barcodeFormat'] }}';
        var settings = {
            output: renderer,
            bgColor: '#FFFFFF',
            color: '#000000',
            barWidth: '1',
            barHeight: '50',
            moduleSize: '5',
            posX: '10',
            posY: '20',
            addQuietZone: '1'
        };
        $('.product_barcode_' + id).html("").show().barcode(value, btype, settings);

    }
</script>
</body>
</html>
