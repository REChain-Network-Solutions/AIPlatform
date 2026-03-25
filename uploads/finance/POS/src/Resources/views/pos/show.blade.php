@if (!empty($sales) && count($sales) > 0)

    <div class="container">
        <div class="row align-items-center mb-4 invoice mt-2">
            <div class="col invoice-details">
                <h1 class="invoice-id h6">{{ $details['pos_id'] }}</h1>
                <div class="date"><b>{{ __('Date') }}: </b>{{ $details['date'] }}</div>
            </div>
            <div class="col d-flex justify-content-end">
                <div class="text-dark mb-0 mt-2"><b>{{ __('Warehouse Name') }}: </b>{!! $details['warehouse']['details'] !!}</div>
            </div>
        </div>
        <div class="row invoice mt-2">
            <div class="col contacts d-flex justify-content-between pb-4">
                <div class="invoice-to">
                    <div class="text-dark h6"><b>{{ __('Billed To :') }}</b></div>
                    {!! $details['customer']['details'] !!}
                </div>
                @if (!empty($details['customer']['shippdetails']))
                    <div class="invoice-to">
                        <div class="text-dark h6"><b>{{ __('Shipped To :') }}</div>
                        {!! $details['customer']['shippdetails'] !!}
                    </div>
                @endif
                <div class="company-details">
                    <div class="text-dark h6"><b>{{ __('From:') }}</b></div>
                    {!! $details['user']['details'] !!}
                </div>
            </div>
        </div>
        <div class="col-12 col-md-12">
            <div class="invoice-table" data-repeater-list="items">
                <div class="table-responsive">
                <table class="table">
                    <thead>
                        <tr>
                            <th class="text-left">{{ __('Items') }}</th>
                            <th>{{ __('Quantity') }}</th>
                            <th class="text-right">{{ __('Price') }}</th>
                            <th class="text-right">{{ __('Tax') }}</th>
                            <th class="text-right">{{ __('Tax Amount') }}</th>
                            <th class="text-right">{{ __('Total') }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        @if (array_key_exists('data', $sales))
                            @foreach ($sales['data'] as $key => $value)
                                <tr>
                                    <td class="cart-summary-table text-left">
                                        {{ $value['name'] }}
                                    </td>
                                    <td class="cart-summary-table">
                                        {{ $value['quantity'] }}
                                    </td>
                                    <td class="text-right cart-summary-table">
                                        {{ $value['price'] }}
                                    </td>
                                    <td class="text-right cart-summary-table">
                                        {!! $value['product_tax'] !!}
                                    </td>
                                    <td class="text-right cart-summary-table">
                                        {{ $value['tax_amount'] }}
                                    </td>
                                    <td class="text-right cart-summary-table">
                                        {{ $value['subtotal'] }}
                                    </td>
                                </tr>
                            @endforeach
                        @endif
                    </tbody>
                    <tfoot>
                        <tr>
                            <td class="">{{ __('Sub Total') }}</td>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td class="text-right">{{ $sales['sub_total'] }}</td>
                        </tr>

                        <tr>
                            <td class="">{{ __('Discount') }}</td>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td class="text-right">{{ $sales['discount'] }}</td>
                        </tr>
                        <tr>
                            <td class="text-left font-weight-bold">{{ __('Total') }}</td>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td class="text-right font-weight-bold">{{ $sales['total'] }}</td>
                        </tr>
                    </tfoot>
                </table>
            </div>
            </div>
            @if ($details['pay'] == 'show')

                    <button class="btn btn-success payment-done-btn rounded mb-3 float-right"
                    data-url="{{ route('pos.printview') }}" data-ajax-popup="true" data-size="sm"
                    data-bs-toggle="tooltip" data-title="{{ __('POS Invoice') }}">
                    {{ __('Cash Payment') }}
                </button>
            @endif
        </div>

    </div>

@endif


<script type="text/javascript" src="{{ asset('js/html2pdf.bundle.min.js') }}"></script>
<script>
    var filename = $('#filename').val()

    function saveAsPDF() {
        var element = document.getElementById('printableArea');
        var opt = {
            margin: 0.3,
            filename: filename,
            image: {
                type: 'jpeg',
                quality: 1
            },
            html2canvas: {
                scale: 4,
                dpi: 72,
                letterRendering: true
            },
            jsPDF: {
                unit: 'in',
                format: 'A2'
            }
        };
        html2pdf().set(opt).from(element).save();
    }

        $(document).on('click', '.payment-done-btn', function(e) {
        e.preventDefault();
        var ele = $(this);

        $.ajax({
            url: "{{ route('pos.data.store') }}",

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
                    $('#carthtml').load(document.URL + ' #carthtml');
                    show_toastr('success', data.success, 'success')
                }
            },
            error: function(data) {
                data = data.responseJSON;
                show_toastr('{{ __('Error') }}', data.error, 'error');
            }

        });
    });
</script>
