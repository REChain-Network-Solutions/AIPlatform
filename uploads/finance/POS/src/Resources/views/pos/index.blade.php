@php
    $admin_settings = getAdminAllSetting();
    $company_settings = getCompanyAllSetting();
    $favicon = isset($company_settings['favicon']) ? $company_settings['favicon'] : (isset($admin_settings['favicon']) ? $admin_settings['favicon'] : 'uploads/logo/favicon.png');
    $color = !empty($company_settings['color']) ? $company_settings['color'] : 'theme-1';
      if (isset($company_settings['color_flag']) && $company_settings['color_flag'] == 'true') {
          $themeColor = 'custom-color';
      } else {
          $themeColor = $color;
      }
@endphp
<!DOCTYPE html>

<html lang="{{ str_replace('_', '-', app()->getLocale()) }}" dir="{{company_setting('site_rtl') == 'on'?'rtl':''}}">

<head>
    <title>
        {{ __('POS') }} |
        {{ isset($company_settings['header_text']) && !empty($company_settings['header_text']) ? $company_settings['header_text']->value : config('app.name', 'WorkDo Crm') }}
    </title>
    <meta name="title" content="{{ isset($admin_settings['meta_title']) && !empty($admin_settings['meta_title']) ? $admin_settings['meta_title'] : 'WOrkdo Dash' }}">
    <meta name="keywords" content="{{ isset($admin_settings['meta_keywords']) && !empty($admin_settings['meta_keywords']) ? $admin_settings['meta_keywords'] : 'WorkDo Dash,SaaS solution,Multi-workspace' }}">
    <meta name="description" content="{{ isset($admin_settings['meta_description']) && !empty($admin_settings['meta_description']) ? $admin_settings['meta_description'] : 'Discover the efficiency of Dash, a user-friendly web application by WorkDo.'}}">

    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="website">
    <meta property="og:url" content="{{ env('APP_URL') }}">
    <meta property="og:title" content="{{ isset($admin_settings['meta_title']) && !empty($admin_settings['meta_title']) ? $admin_settings['meta_title'] : 'WOrkdo Dash' }}">
    <meta property="og:description" content="{{ isset($admin_settings['meta_description']) && !empty($admin_settings['meta_description']) ? $admin_settings['meta_description'] : 'Discover the efficiency of Dash, a user-friendly web application by WorkDo.'}} ">
    <meta property="og:image" content="{{ get_file( (isset($admin_settings['meta_image']) && !empty($admin_settings['meta_image'])) ? (check_file($admin_settings['meta_image'])) ?  $admin_settings['meta_image'] : 'uploads/meta/meta_image.png' : 'uploads/meta/meta_image.png'  ) }}{{'?'.time() }}">

    <!-- Twitter -->
    <meta property="twitter:card" content="summary_large_image">
    <meta property="twitter:url" content="{{ env('APP_URL') }}">
    <meta property="twitter:title" content="{{ isset($admin_settings['meta_title']) && !empty($admin_settings['meta_title']) ? $admin_settings['meta_title'] : 'WOrkdo Dash' }}">
    <meta property="twitter:description" content="{{ isset($admin_settings['meta_description']) && !empty($admin_settings['meta_description']) ? $admin_settings['meta_description'] : 'Discover the efficiency of Dash, a user-friendly web application by WorkDo.'}} ">
    <meta property="twitter:image" content="{{ get_file( (isset($admin_settings['meta_image']) && !empty($admin_settings['meta_image'])) ? (check_file($admin_settings['meta_image'])) ?  $admin_settings['meta_image'] : 'uploads/meta/meta_image.png' : 'uploads/meta/meta_image.png'  ) }}{{'?'.time() }}">

    <meta name="author" content="Workdo.io">

    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <link rel="icon" href="{{ check_file($favicon) ? get_file($favicon) : get_file('uploads/logo/favicon.png') }}{{ '?' . time() }}" type="image/x-icon" />
    <link rel="stylesheet" href="{{ asset('assets/fonts/tabler-icons.min.css') }}">
    <link rel="stylesheet" href="{{ asset('packages/workdo/Pos/src/Resources/assets/css/site.css') }}" id="stylesheet">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <!-- font css -->
    <link rel="stylesheet" href="{{ asset('assets/fonts/tabler-icons.min.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/fonts/feather.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/fonts/fontawesome.css') }}">
    <link rel="stylesheet" href="{{ asset('assets/fonts/material.css') }}">

    <style>
        :root {
            --color-customColor: <?= $color ?>;
        }
        /* Chrome, Safari, Edge, Opera */
        input[type=number][name="quantity"]::-webkit-outer-spin-button,
        input[type=number][name="quantity"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        /* Firefox */
        input[type=number][name="quantity"] {
            -moz-appearance: textfield;
        }
    </style>
    <link rel="stylesheet" href="{{ asset('css/custome.css') }}">
    <link rel="stylesheet" href="{{ asset('css/custom-color.css') }}">

    <!--bootstrap switch-->
    <link rel="stylesheet" href="{{ asset('assets/css/plugins/bootstrap-switch-button.min.css') }}">

    <!-- vendor css -->
    @if (isset($company_settings['site_rtl']) && $company_settings['site_rtl'] == 'on')
        <link rel="stylesheet" href="{{ asset('assets/css/style-rtl.css') }}">
    @endif
    @if (isset($company_settings['cust_darklayout']) && $company_settings['cust_darklayout'] == 'on')
        <link rel="stylesheet" href="{{ asset('assets/css/style-dark.css') }}">
    @else
        <link rel="stylesheet" href="{{ asset('assets/css/style.css') }}" id="main-style-link">
    @endif

    <link rel="stylesheet" href="{{ asset('assets/css/customizer.css') }}">
    <link rel="stylesheet" href="{{ asset('packages/workdo/Pos/src/Resources/assets/css/custom.css') }}" id="main-style-link">
    <style>

    </style>
    @stack('css')
</head>

<body class="{{ isset($themeColor) ? $themeColor : 'theme-1' }}">
    <!-- [ Main Content ] start -->
    <div class="container-fluid px-2">
        <?php $lastsegment = request()->segment(count(request()->segments())); ?>
        <div class="row">
            <div class="col-12">
                <div class="mt-2 pos-top-bar bg-color d-flex align-items-center justify-content-between bg-primary gap-2">
                   <div class="d-flex flex-wrap align-items-center gap-2">
                        <div class="logo">
                            <a href="{{ route('dashboard')}}" class="b-brand">
                                <img src="{{ get_file(sidebar_logo()) }}{{ '?' . time() }}"
                                    alt="" style="max-width: 100px" />
                            </a>
                        </div>
                   </div>
                    <a href="{{ route('dashboard') }}" class="text-white"><i class="ti ti-home"
                            style="font-size: 20px;"></i> </a>
                </div>
            </div>
        </div>

        <div class="mt-2 row row-gap">
            <div class="col-xl-8">
                <div class="sop-card card">
                    <div class="card-body p-2">
                        <div class="right-content">
                            <div class="search-bar-left mb-3">
                                <form>
                                    <div class="input-group bg-white rounded">
                                        <input id="searchproduct" type="text" data-url="{{ route('search.products') }}"
                                            placeholder="{{ __('Search Product') }}"
                                            class="form-control pr-4 shadow-none">
                                        <span class="input-group-text"><i class="ti ti-search"></i></span>
                                    </div>
                                </form>
                            </div>
                            <div class="button-list b-bottom catgory-pad p-2 card">
                                <div class="form-row top-10-scroll m-0" id="categories-listing" style="height: 100px" >
                                </div>
                            </div>
                            <div class="product-body-nop">
                                <div class="form-row" id="product-listing">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-4 ps-xl-0">
                <div class="card m-0">
                    <div class="card-header p-2">
                        <div class="row row-gap">
                            <div class="col-md-6">
                                {{ Form::select('customer_id', $customers,$customer, array('class' => 'form-control select customer_select','id'=>'customer','required'=>'required')) }}
                                {{ Form::hidden('vc_name_hidden', '',['id' => 'vc_name_hidden']) }}
                            </div>
                            <div class="col-md-6">
                                {{ Form::select('warehouse_id', $warehouses,$warehouseId, array('class' => 'form-control select warehouse_select ','id'=>'warehouse','required'=>'required')) }}
                                {{ Form::hidden('warehouse_name_hidden', '',['id' => 'warehouse_name_hidden']) }}
                                {{ Form::hidden('quotation_id', $id,['id' => 'quotation_id' , 'class'=>"quotation"]) }}
                            </div>
                        </div>
                    </div>
                    <div class="card-body carttable cart-product-list carttable-scroll p-3" id="carthtml">

                        @php $total = 0 @endphp

                        <div class="table-responsive">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th></th>
                                        <th class="text-left">{{ __('Name') }}</th>
                                        <th class="text-center">{{ __('QTY') }}</th>
                                        <th>{{ 'Tax' }}</th>
                                        <th class="text-center">{{ __('Price') }}</th>
                                        <th class="text-center">{{ __('Sub Total') }}</th>
                                        <th></th>
                                    </tr>
                                </thead>
                                @if (module_is_active('ProductService'))
                                    <tbody id="tbody">
                                        @php
                                                $ids = \App\Models\WarehouseProduct::All()->pluck('product_id')->toArray();
                                        @endphp
                                        @if (session($lastsegment) && !empty(session($lastsegment)) && count(session($lastsegment)) > 0)
                                            @foreach (session($lastsegment) as $id => $details)
                                            @if (in_array($details['id'], $ids))
                                                @php

                                                    $product = \workdo\ProductService\Entities\ProductService::find($details['id']);
                                                    $image_url = !empty($product) && check_file($product->image) && get_file($product->image) ? $product->image : asset('packages/workdo/ProductService/src/Resources/assets/image/img01.jpg');
                                                    $total += $details['subtotal'];
                                                @endphp

                                                <tr data-product-id="{{ $id }}" id="product-id-{{ $id }}">
                                                    <td class="cart-images">
                                                        <img alt="Image placeholder" src="{{ get_file($image_url) }}"
                                                            class="card-image avatar rounded-circle-sale shadow hover-shadow-lg">
                                                    </td>
                                                    <td class="name">{{ $details['name'] }}</td>
                                                    <td>
                                                        <span class="quantity buttons_added">
                                                                <button class="minus">
                                                                    <svg width="20px" height="20px" viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:sketch="http://www.bohemiancoding.com/sketch/ns">
                                                                        <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd" sketch:type="MSPage">
                                                                            <g id="Icon-Set-Filled" sketch:type="MSLayerGroup" transform="translate(-518.000000, -1089.000000)" fill="#000000">
                                                                                <path d="M540,1106 L528,1106 C527.447,1106 527,1105.55 527,1105 C527,1104.45 527.447,1104 528,1104 L540,1104 C540.553,1104 541,1104.45 541,1105 C541,1105.55 540.553,1106 540,1106 L540,1106 Z M534,1089 C525.163,1089 518,1096.16 518,1105 C518,1113.84 525.163,1121 534,1121 C542.837,1121 550,1113.84 550,1105 C550,1096.16 542.837,1089 534,1089 L534,1089 Z" id="minus-circle" sketch:type="MSShapeGroup">
                                                                            </path>
                                                                            </g>
                                                                        </g>
                                                                    </svg>
                                                                </button>                                                            <input type="text" step="1" name="quantity"
                                                                title="{{ __('Quantity') }}" class="input-number"
                                                                data-url="{{ url('update-cart/') }}"
                                                                data-id="{{ $id }}" min="1"
                                                                max="10000"
                                                                value="{{ $details['quantity'] }}">
                                                                <button class="plus">
                                                                    <svg width="20px" height="20px" viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:sketch="http://www.bohemiancoding.com/sketch/ns">
                                                                        <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd" sketch:type="MSPage">
                                                                            <g id="Icon-Set-Filled" sketch:type="MSLayerGroup" transform="translate(-466.000000, -1089.000000)" fill="#000000">
                                                                                <path d="M488,1106 L483,1106 L483,1111 C483,1111.55 482.553,1112 482,1112 C481.447,1112 481,1111.55 481,1111 L481,1106 L476,1106 C475.447,1106 475,1105.55 475,1105 C475,1104.45 475.447,1104 476,1104 L481,1104 L481,1099 C481,1098.45 481.447,1098 482,1098 C482.553,1098 483,1098.45 483,1099 L483,1104 L488,1104 C488.553,1104 489,1104.45 489,1105 C489,1105.55 488.553,1106 488,1106 L488,1106 Z M482,1089 C473.163,1089 466,1096.16 466,1105 C466,1113.84 473.163,1121 482,1121 C490.837,1121 498,1113.84 498,1105 C498,1096.16 490.837,1089 482,1089 L482,1089 Z" id="plus-circle" sketch:type="MSShapeGroup">
                                                                                </path>
                                                                            </g>
                                                                        </g>
                                                                    </svg>
                                                                </button>                                                        </span>
                                                    </td>

                                                    <td>
                                                        @if (!empty($product->tax_id))
                                                            @php
                                                                $taxes = \workdo\Pos\Entities\Pos::tax($product->tax_id);
                                                            @endphp

                                                            @foreach ($taxes as $tax)
                                                                <span
                                                                    class="badge bg-primary">{{ $tax->name . ' (' . $tax->rate . '%)' }}</span>
                                                                <br>
                                                            @endforeach
                                                        @else
                                                            -
                                                        @endif
                                                    </td>
                                                    <td class="price text-right">
                                                        {{ currency_format_with_sym($details['price']) }}</td>
                                                    <td class="col-sm-3 mt-2">
                                                        <span
                                                            class="subtotal">{{ currency_format_with_sym($details['subtotal']) }}</span>
                                                    </td>
                                                    <td class="col-sm-2 mt-2">
                                                        <a href="#"
                                                            class="action-btn  show_confirm-pos"
                                                            data-confirm="{{ __('Are You Sure?') }}"
                                                            data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                            data-confirm-yes="delete-form-{{ $id }}"
                                                            title="{{ __('Delete') }}"
                                                            data-id="{{ $id }}">
                                                            <i class="ti ti-trash bg-danger text-white mx-3 btn btn-sm"
                                                                title="{{ __('Delete') }}"></i>
                                                        </a>
                                                        {!! Form::open(['method' => 'delete', 'url' => ['remove-from-cart'], 'id' => 'delete-form-' . $id]) !!}
                                                        <input type="hidden" name="session_key"
                                                            value="{{ $lastsegment }}">
                                                        <input type="hidden" name="id"
                                                            value="{{ $id }}">
                                                        {!! Form::close() !!}
                                                    </td>
                                                </tr>
                                            @endif
                                            @endforeach
                                        @else
                                            <tr class="text-center no-found">
                                                <td colspan="7">{{ __('No Data Found.!') }}</td>
                                            </tr>
                                        @endif
                                    </tbody>
                                @endif
                            </table>
                        </div>
                        <div class="total-section" @if(isset($company_settings['site_rtl']) && $company_settings['site_rtl'] == 'on') style="background: #0000" @endif>
                            <div class="sub-total">
                                <div class="row row-gap align-items-center">
                                    <div class="col-sm-6">
                                        <div class="d-flex text-end justify-content-end align-items-center">
                                            <div class="input-group">
                                                <span class="input-group-text g-1 bg-transparent currency-symbol rounded-start">{{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '$' }}</span>
                                                {{ Form::number('discount', null, ['class' => 'form-control discount rounded-end', 'required' => 'required', 'placeholder' => __('Discount')]) }}
                                                {{ Form::hidden('discount_hidden', '', ['id' => 'discount_hidden']) }}
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-sm-6">
                                        <div class="d-flex text-end justify-content-end mb-2">
                                            <h6 class="mb-0 text-dark">{{ __('Sub Total') }} :</h6>
                                            <h6 class="mb-0 text-gray-800" id="displaytotal">
                                                {{ currency_format_with_sym($total) }}</h6>
                                        </div>
                                        <div class="d-flex align-items-center justify-content-end">
                                            <h6 class="mb-0">{{ __('Total') }} :</h6>
                                            <h6 class="mb-0 totalamount">{{ currency_format_with_sym($total) }}</h6>
                                        </div>
                                    </div>
                                </div>
                                <div class="d-flex align-items-center justify-content-between pt-3 gap-2" id="btn-pur">
                                    @if (module_is_active('ProductService'))
                                        <button type="button" class="btn btn-primary pay-btn m-0" data-ajax-popup="true"
                                            data-size="lg" data-align="centered"
                                            data-url="{{ route('pos.create') }}"
                                            data-title="{{ __('POS Invoice') }}"
                                            @if (session($lastsegment) && !empty(session($lastsegment)) && count(session($lastsegment)) > 0) @else disabled="disabled" @endif>
                                            {{ __('PAY') }}
                                        </button>
                                    @else
                                        <button type="button" class="btn btn-primary" data-ajax-popup="true"
                                            data-size="lg" data-align="centered"
                                            data-url="{{ route('pos.create') }}"
                                            data-title="{{ __('POS Invoice') }}" disabled="disabled">
                                            {{ __('PAY') }}
                                        </button>
                                    @endif
                                    <div class="tab-content btn-empty text-end">
                                        <a href="#" class="btn btn-danger show_confirm-pos m-0"
                                            data-toggle="tooltip" data-original-title="{{ __('Empty Cart') }}"
                                            data-confirm="{{ __('Are You Sure?') }}"
                                            data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                            data-confirm-yes="delete-form-emptycart"
                                            @if (session($lastsegment) && !empty(session($lastsegment)) && count(session($lastsegment)) > 0) @else style="pointer-events: none; cursor: default; opacity: 0.65;" @endif>
                                            {{ __('Empty Cart') }}
                                        </a>
                                        {!! Form::open(['method' => 'post', 'url' => ['empty-cart'], 'id' => 'delete-form-emptycart']) !!}
                                        <input type="hidden" name="session_key" value="{{ $lastsegment }}"
                                            id="empty_cart">

                                        {!! Form::close() !!}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    </div>

    <div class="modal fade" id="commonModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel"
        aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="exampleModalLabel"></h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" id="reload_page" aria-label="Close"></button>
                </div>
                <div class="body">
                </div>
            </div>
        </div>
    </div>
    <div class="position-fixed top-0 end-0 p-3" style="z-index: 99999">
        <div id="liveToast" class="toast text-white  fade" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body"> </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"
                    aria-label="Close"></button>
            </div>
        </div>
    </div>

    <!-- Required Js -->
    <script src="{{ asset('js/jquery.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/popper.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/perfect-scrollbar.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/bootstrap.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/feather.min.js') }}"></script>
    <script src="{{ asset('assets/js/dash.js') }}"></script>
    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/moment.min.js') }}"></script>

    <script src="{{ asset('assets/js/plugins/bootstrap-switch-button.min.js') }}"></script>

    <script src="{{ asset('assets/js/plugins/sweetalert2.all.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/simple-datatables.js') }}"></script>

    <!-- Apex Chart -->
    <script src="{{ asset('assets/js/plugins/apexcharts.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/main.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/choices.min.js') }}"></script>
    <script src="{{ asset('assets/js/plugins/flatpickr.min.js') }}"></script>



    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/jscolor.js') }}"></script>
    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/custom.js') }}"></script>

    @if ($message = Session::get('success'))
        <script>
            show_toastr('success', '{!! $message !!}');
        </script>
    @endif
    @if ($message = Session::get('error'))
        <script>
            show_toastr('error', '{!! $message !!}');
        </script>
    @endif
    @if(isset($admin_settings['enable_cookie']) && $admin_settings['enable_cookie'] == 'on')
    @include('layouts.cookie_consent')
@endif
    @stack('scripts')

    <script src="{{ asset('js/jquery-ui.min.js') }}"></script>

    <script>
        $.ajaxSetup({
            headers: {
                'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
            }
        });


        $(document).ready(function() {

            $("#vc_name_hidden").val($('.customer_select').val());
            $("#warehouse_name_hidden").val($('.warehouse_select').val());
            $("#discount_hidden").val($('.discount').val());
            $( "#quotation_id").val($('.quotation').val());

            $(function() {
                getProductCategories();

            });

            if ($('#searchproduct').length > 0) {
                var url = $('#searchproduct').data('url');
                var ware_id = $("#warehouse").val();
                var cat = $(this).data('cat-id');
                searchProducts(url, '', cat, ware_id);
            }

            $('#warehouse').change(function() {
                var ware_id = $("#warehouse").val();
                searchProducts(url, '', '0', ware_id);
            });

            $('.customer_select').change(function() {
                $("#vc_name_hidden").val($(this).val());
            });
            $('.warehouse_select').change(function() {
                $("#warehouse_name_hidden").val($(this).val());

                var session_key = $("#empty_cart").val();
                $.ajax({
                    type: 'POST',
                    url: '{{ route('warehouse-empty-cart') }}',
                    data: {
                        'session_key': session_key
                    },
                    success: function(data) {
                        $("#tbody").empty();
                        $('.pay-btn').attr('disabled','disabled');
                        $('#btn-pur .btn-empty a').attr('style','pointer-events: none; cursor: default; opacity: 0.65;');
                        $("#tbody").html(
                            '<tr class="text-center no-found"><td colspan="7">{{ __('No Data Found.!') }}</td></tr>'
                            );

                    }
                });
            });

            $(document).on('click', '#clearinput', function(e) {
                var IDs = [];
                $(this).closest('div').find("input").each(function() {
                    IDs.push('#' + this.id);
                });
                $(IDs.toString()).val('');
            });

            $(document).on('keyup', 'input#searchproduct', function() {
                var url = $(this).data('url');
                var value = this.value;
                var cat = $('.cat-active').children().data('cat-id');
                var ware_id = $("#warehouse").val();

                searchProducts(url, value, cat,ware_id);
            });


            function searchProducts(url, value, cat_id, war_id = '0') {

                $.ajax({
                    type: 'GET',
                    url: url,
                    data: {
                        'search': value,
                        'cat_id': cat_id,
                        'war_id': war_id,
                        'session_key': '{{ $lastsegment}}'

                    },
                    success: function(data) {
                        $("#product-listing").empty();
                        $('#product-listing').html(data);
                    }
                });
            }

            function getProductCategories() {
                $.ajax({
                    type: 'GET',
                    url: '{{ route('product.categories') }}',
                    success: function(data) {

                        $('#categories-listing').html(data);
                    }
                });
            }

            $(document).on('click', '.toacart', function() {
                var sum = 0;
                $.ajax({
                    url: $(this).data('url'),

                    success: function(data) {

                        if (data.code == '200') {

                            $('#displaytotal').text(addCommas(data.product.subtotal));
                            $('.totalamount').text(addCommas(data.product.subtotal));
                            if ('carttotal' in data) {
                                $.each(data.carttotal, function(key, value) {
                                    $('#product-id-' + value.id + ' .subtotal').text(
                                        addCommas(value.subtotal));
                                    sum += value.subtotal;
                                });
                                $('#displaytotal').text(addCommas(sum));
                                $('.totalamount').text(addCommas(sum));

                                $('.discount').val('');
                            }

                            $('#tbody').append(data.carthtml);
                            $('.no-found').addClass('d-none');
                            $('.carttable #product-id-' + data.product.id +
                                ' input[name="quantity"]').val(data.product.quantity);
                            $('#btn-pur button').removeAttr('disabled');
                            $('#btn-pur .btn-empty a').removeAttr('style');
                            $('.btn-empty button').addClass('btn-clear-cart');
                        }
                    },
                    error: function(data) {
                        data = data.responseJSON;
                        show_toastr('{{ __('Error') }}', data.error, 'error');
                    }
                });
            });

            $(document).on('change keyup', '#carthtml input[name="quantity"]', function (e) {
            e.preventDefault();
            var ele = $(this);
            var sum = 0;
            var quantity = ele.closest('span').find('input[name="quantity"]').val();
            var discount = $('.discount').val();
            var session_key = $('#empty_cart').val();

            if(quantity != "" && quantity != 0){
                $.ajax({
                    url: ele.data('url'),
                    method: "patch",
                    data: {
                        id: ele.attr("data-id"),
                        quantity: quantity,
                        discount:discount,
                        session_key: session_key
                    },
                    success: function (data) {

                        if (data.code == '200') {

                            if (quantity == 0) {
                                ele.closest(".row").hide(250, function () {
                                    ele.closest(".row").remove();
                                });
                                if (ele.closest(".row").is(":last-child")) {
                                    $('#btn-pur button').attr('disabled', 'disabled');
                                    $('.btn-empty button').removeClass('btn-clear-cart');
                                }
                            }

                            $.each(data.product, function (key, value) {
                                sum += value.subtotal;
                                $('#product-id-' + value.id + ' .subtotal').text(addCommas(value.subtotal));
                            });

                            $('#displaytotal').text(addCommas(sum));
                            if(discount <= sum){
                                $('.totalamount').text(data.discount);
                            }
                            else{
                                $('.totalamount').text(addCommas(0));
                            }
                        }
                    },
                    error: function (data) {
                        data = data.responseJSON;
                        show_toastr('{{ __("Error") }}', data.error, 'error');
                    }
                });
            }


        });

            $(document).on('click', '.remove-from-cart', function(e) {
                e.preventDefault();
                var ele = $(this);
                var sum = 0;

                if (confirm('{{ __("Are you sure?") }}')) {
                    ele.closest(".row").hide(250, function() {
                        ele.closest(".row").parent().parent().remove();
                    });
                    if (ele.closest(".row").is(":last-child")) {
                        $('#btn-pur button').attr('disabled', 'disabled');
                        $('.btn-empty button').removeClass('btn-clear-cart');
                    }
                    $.ajax({
                        url: ele.data('url'),
                        method: "DELETE",
                        data: {
                            id: ele.attr("data-id"),
                        },
                        success: function(data) {
                            if (data.code == '200') {

                                $.each(data.product, function(key, value) {
                                    sum += value.subtotal;
                                    $('#product-id-' + value.id + ' .subtotal').text(
                                        addCommas(value.subtotal));
                                });

                                $('#displaytotal').text(addCommas(sum));

                                toastrs('Success', data.success, 'success')
                            }
                        },
                        error: function(data) {
                            data = data.responseJSON;
                            show_toastr('{{ __('Error') }}', data.error, 'error');
                        }
                    });
                }
            });

            $(document).on('click', '.btn-clear-cart', function(e) {
                e.preventDefault();

                if (confirm('{{ __("Remove all items from cart?") }}')) {

                    $.ajax({
                        url: $(this).data('url'),
                        data: {
                            session_key: session_key
                        },
                        success: function(data) {
                            location.reload();
                        },
                        error: function(data) {
                            data = data.responseJSON;
                            show_toastr('{{ __('Error') }}', data.error, 'error');
                        }
                    });
                }
            });

            $(document).on('click', '.btn-done-payment', function(e) {
                e.preventDefault();
                var ele = $(this);

                $.ajax({
                    url: ele.data('url'),

                    method: 'GET',
                    data: {
                        vc_name: $('#vc_name_hidden').val(),
                        warehouse_name: $('#warehouse_name_hidden').val(),
                        discount: $('#discount_hidden').val(),
                        quotation_id : $('#quotation_id').val(),

                    },
                    beforeSend: function() {
                        ele.remove();
                    },
                    success: function(data) {
                        if (data.code == 200) {
                            show_toastr('Success', data.success, 'success');
                        }
                    },
                    error: function(data) {
                        data = data.responseJSON;
                        show_toastr('{{ __('Error') }}', data.error, 'error');
                    }

                });

            });
            $(document).on('click', '#reload_page', function(e) {
                e.preventDefault();

                var ele = $(this);
                setTimeout(function() {
                    window.location.reload();
                }, 1000);
            }, );

            $(document).on('click', '.category-select', function(e) {
                var cat = $(this).data('cat-id');
                var white = 'text-white';
                var dark = 'text-dark';
                $('.category-select').parent().removeClass('cat-active');
                $('.category-select').children().removeClass('btn-primary');
                $(this).children().addClass('btn-primary');
                $('.category-select').find('.card-title').removeClass('text-white').addClass('text-dark');
                $('.category-select').find('.card-title').parent().removeClass('text-white').addClass('text-dark');
                $(this).find('.card-title').removeClass('text-dark').addClass('text-white');
                $(this).find('.card-title').parent().removeClass('text-dark').addClass('text-white');
                $(this).parent().addClass('cat-active');
                var url = '{{ route('search.products') }}'
                var ware_id = $("#warehouse").val();
                searchProducts(url, '', cat, ware_id);


            });

            $(document).on('keyup', '.discount', function () {
                var discount = $('.discount').val();
                var total = $('#displaytotal').html().replace(/[^0-9.]/gi, ''); // Assuming the displaytotal element contains the total amount

                $("#discount_hidden").val(discount);

                $.ajax({
                    url: "{{ route('cartdiscount') }}",
                    method: 'POST',
                    data: {
                        discount: discount,
                    },
                    success: function (data) {
                        console.log(data,discount,total);
                        if (parseFloat(discount) <= parseFloat(total) || discount == null || discount == '') {
                            $('.totalamount').text(data.total);
                        } else {
                            console.log("asd");
                            $('.totalamount').text(addCommas(0));
                        }
                    },
                    error: function (data) {
                        data = data.responseJSON;
                        show_toastr('{{ __('Error') }}', data.error, 'error');
                    }
                });
            });
        });
    </script>


    <script>
        var site_currency_symbol_position = "{{isset($company_settings['site_currency_symbol_position']) ? $company_settings['site_currency_symbol_position'] : 'pre'}}";
        var site_currency_symbol = "{{isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '$'}}";
    </script>


</body>

</html>
