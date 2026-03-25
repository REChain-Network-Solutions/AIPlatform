@php
    $company_settings = getCompanyAllSetting();
@endphp
<h5 class="h4 d-inline-block font-weight-400 mb-4 pro_name">{{ __('Project') }}</h5>
{{ Form::hidden('itemTaxRate', null, ['class' => 'form-control itemTaxRate']) }}
    <div class="card repeater" @if ($acction == 'edit') data-value='{!! json_encode($invoice->items) !!}' @endif>
        <div class="item-section p-3 pb-0">
            <div class="row justify-content-between align-items-center">
                <div class="col-md-12 d-flex align-items-center justify-content-md-end px-4">
                    <a href="#" data-repeater-create="" class="btn btn-primary tax_get mr-2" data-toggle="modal"
                        data-target="#add-bank">
                        <i class="ti ti-plus"></i> {{ __('Add Item') }}
                    </a>
                </div>
            </div>
        </div>
        <div class="card-body table-border-style">
            <div class="table-responsive">
                <table class="table  mb-0 table-custom-style" data-repeater-list="items" id="sortable-table">
                    <thead>
                        <tr>
                            <th>{{ __('Items') }}</th>
                            <th>{{ __('Price') }} </th>
                            <th>{{ __('Discount') }}</th>
                            <th width="200px">{{ __('Tax') }} (%)</th>
                            <th class="text-end">{{ __('Amount') }} <br><small
                                    class="text-danger font-weight-bold">{{ __('After discount & tax') }}</small></th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody class="ui-sortable" data-repeater-item>
                        <tr>
                            <td width="25%" class="form-group pt-0">
                                {{ Form::hidden('id', null, ['class' => 'form-control id']) }}
                                <select name="product_id" class="form-control product_id item js-searchBox" required>
                                    <option value="0">--</option>
                                    @foreach ($tasks as $key => $task)
                                        <option value="{{ $task->id }}">{{ $task->title }}</option>
                                    @endforeach
                                </select>
                            </td>
                            <td>
                                <div class="form-group price-input input-group search-form mb-0" style="width: 160px">
                                    {{ Form::text('price', '', ['class' => 'form-control price', 'required' => 'required', 'placeholder' => __('Price'), 'required' => 'required']) }}
                                    <span
                                        class="input-group-text bg-transparent">{{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '' }}</span>
                                </div>
                            </td>
                            {{ Form::hidden('quantity', 1, ['class' => 'form-control quantity', 'required' => 'required', 'placeholder' => __('Qty'), 'required' => 'required']) }}
                            <td>
                                <div class="form-group price-input input-group search-form mb-0" style="width: 160px">
                                    {{ Form::text('discount', '', ['class' => 'form-control discount', 'placeholder' => __('Discount')]) }}
                                    <span
                                        class="input-group-text bg-transparent">{{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '' }}</span>
                                </div>
                            </td>
                            <td>
                                <div class="form-group mb-0">
                                    <div class="input-group">
                                        <div class="taxes"></div>
                                        {{ Form::hidden('tax', null, ['class' => 'form-control tax']) }}
                                        {{ Form::hidden('itemTaxPrice', '', ['class' => 'form-control itemTaxPrice']) }}
                                        {{ Form::hidden('itemTaxRate', '', ['class' => 'form-control itemTaxRate']) }}
                                    </div>
                                </div>
                            </td>
                            <td class="text-end amount">{{ __('0.00') }}</td>
                            <td>
                                <div class="action-btn ms-2 float-end" data-repeater-delete>
                                    <a href="#!"
                                        class="mx-3 btn btn-sm d-inline-flex align-items-center m-2 p-2 bg-danger">
                                        <i class="ti ti-trash text-white" data-bs-toggle="tooltip"
                                            data-bs-original-title="{{ __('Delete') }}" ></i>
                                    </a>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td colspan="2">
                                <div class="form-group mb-0">
                                    {{ Form::textarea('description', null, ['class' => 'form-control', 'rows' => '2', 'placeholder' => __('Description')]) }}
                                </div>
                            </td>
                            <td colspan="5"></td>
                        </tr>
                    </tbody>
                    <tfoot>
                        <tr>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td><strong>{{ __('Sub Total') }}
                                    ({{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '' }})</strong>
                            </td>
                            <td class="text-end subTotal">{{ __('0.00') }}</td>
                        </tr>
                        <tr>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td><strong>{{ __('Discount') }}
                                    ({{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '' }})</strong>
                            </td>
                            <td class="text-end totalDiscount">{{ __('0.00') }}</td>
                        </tr>
                        <tr>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td><strong>{{ __('Tax') }}
                                    ({{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '' }})</strong>
                            </td>
                            <td class="text-end totalTax">{{ __('0.00') }}</td>
                        </tr>
                        <tr>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td>&nbsp;</td>
                            <td class="blue-text"><strong>{{ __('Total Amount') }}
                                    ({{ isset($company_settings['defult_currancy_symbol']) ? $company_settings['defult_currancy_symbol'] : '' }})</strong>
                            </td>
                            <td class="text-end totalAmount blue-text">{{ __('0.00') }}</td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        </div>
    </div>

<script>
    var allTasks = @json($tasks);

    function refreshTaskSelectOptions() {
        const selectedTaskIds = $('.product_id').map(function () {
            return $(this).val();
        }).get().filter(val => val && val !== "0");

        $('.product_id').each(function () {
            const $select = $(this);
            const currentValue = $select.val();

            $select.empty().append(`<option value="0">--</option>`);

            allTasks.forEach(task => {
                const isSelectedInOther = selectedTaskIds.includes(task.id.toString()) && currentValue !== task.id.toString();
                if (!isSelectedInOther) {
                    $select.append(`<option value="${task.id}" ${currentValue == task.id ? 'selected' : ''}>${task.title}</option>`);
                }
            });
        });
    }    

    var selector = "body";
    if ($(selector + " .repeater").length) {
        var $dragAndDrop = $("body .repeater tbody").sortable({
            handle: '.sort-handler'
        });
        var $repeater = $(selector + ' .repeater').repeater({
            initEmpty: false,
            defaultValues: {
                'status': 1
            },
            show: function() {
                $(this).slideDown();
                $(this).find('input.itemTaxRate').val($('.itemTaxRate').val()); 
                var file_uploads = $(this).find('input.multi');
                if (file_uploads.length) {
                    $(this).find('input.multi').MultiFile({
                        max: 3,
                        accept: 'png|jpg|jpeg',
                        max_size: 2048
                    });
                }
                // for item SearchBox ( this function is  custom Js )
                refreshTaskSelectOptions();
                JsSearchBox();
            },
            hide: function(deleteElement) {
                var $row = $(this); // The current row
                var id = $row.find('.id').val(); // Get the item ID

               // Call the global delete function
                deleteInvoiceItem($row, id, function() {      
                    // Call the global delete function
                    var totalItemTaxPrice = 0;
                    var itemTaxPriceInput = $('.itemTaxPrice');
                    for (var j = 0; j < itemTaxPriceInput.length; j++) {
                        totalItemTaxPrice += parseFloat(itemTaxPriceInput[j].value);
                    }
                    var totalItemPrice = 0;
                    var inputs_quantity = $(".quantity");
                    var priceInput = $('.price');
                    for (var j = 0; j < priceInput.length; j++) {
                        totalItemPrice += (parseFloat(priceInput[j].value) * parseFloat(inputs_quantity[j].value));
                    }
                    var inputs = $(".amount");
                    var subTotal = 0;
                    for (var i = 0; i < inputs.length; i++) {
                        subTotal = parseFloat(subTotal) + parseFloat($(inputs[i]).html());
                    }
                    var totalItemDiscountPrice = 0;
                    var itemDiscountPriceInput = $('.discount');
                    for (var k = 0; k < itemDiscountPriceInput.length; k++) {
                        if (itemDiscountPriceInput[k].value == '') {
                            itemDiscountPriceInput[k].value = parseFloat(0);
                        }
                        totalItemDiscountPrice += parseFloat(itemDiscountPriceInput[k].value);
                    }
                    $('.subTotal').html(totalItemPrice.toFixed(2));
                    $('.totalTax').html(totalItemTaxPrice.toFixed(2));
                    $('.totalAmount').html((parseFloat(subTotal)).toFixed(2));
                    $('.totalDiscount').html(totalItemDiscountPrice.toFixed(2));
                });                
            },
            ready: function(setIndexes) {
                $dragAndDrop.on('drop', setIndexes);
            },
            isFirstItemUndeletable: true
        });
        var value = $(selector + " .repeater").attr('data-value');
        if (typeof value != 'undefined' && value.length != 0) {
            value = JSON.parse(value);
            $repeater.setList(value);
        }

    }
</script>
<script>
    $(document).on('change', '.product_type', function() {
        ProductType($(this));
    });

    function ProductType(data, id = null, type = null) {
        var product_type = data.val();
        var selector = data;
        var itemSelect = selector.parent().parent().find('.product_id.item').attr('name');


        $.ajax({
            url: '{{ route('get.item') }}',
            type: 'POST',
            data: {
                "product_type": product_type,
                "_token": "{{ csrf_token() }}",
            },
            beforeSend: function() {
                $(".loader-wrapper").removeClass('d-none');
            },
            success: function(data) {
                selector.parent().parent().find('.product_id').empty();
                var product_select = `<select class="form-control product_id item js-searchBox" name="${itemSelect}"
                                        placeholder="Select Item" data-url="{{ route('invoice.product') }}" required = 'required'>
                                        </select>`;
                selector.parent().parent().find('.product_div').html(product_select);

                selector.parent().parent().find('.product_id').append(
                    '<option value="0"> {{ __('Select Item') }} </option>');
                $.each(data, function(key, value) {
                    var selected = (key == id) ? 'selected' : '';
                    selector.parent().parent().find('.product_id').append('<option value="' + key +
                        '" ' + selected + '>' + value + '</option>');
                });
                if (type == 'edit') {
                    changeItem(selector.parent().parent().find('.product_id'));
                } else {
                    items(selector.parent().parent().find('.product_id'));
                }
                // Initialize your searchBox here if needed
                selector.parent().parent().find(".js-searchBox").searchBox({
                    elementWidth: '250'
                });
                selector.parent().parent().find('.unit.input-group-text').text("");
                $(".loader-wrapper").addClass('d-none');
            }
        });
    }
</script>

@if ($acction == 'edit')
    <script>
        $(document).ready(function() {

            var value = $(selector + " .repeater").attr('data-value');
            var type = '{{ $type }}';
            if (typeof value != 'undefined' && value.length != 0) {
                value = JSON.parse(value);
                $repeater.setList(value);
                for (var i = 0; i < value.length; i++) {
                    var tr = $('#sortable-table .id[value="' + value[i].id + '"]').parent();
                    tr.find('.item').val(value[i].product_id);
                    if (type == 'product') {
                        var element = tr.find('.product_type');
                        var product_id = value[i].product_id;
                        ProductType(element, product_id, 'edit');
                        changeItem(tr.find('.item'));
                    }
                }
            }
            const elementsToRemove = document.querySelectorAll('.bs-pass-para.repeater-action-btn');
            if (elementsToRemove.length > 0) {
                elementsToRemove[0].remove();
            }
        });
    </script>

    <script>
        $(document).ready(function() {
            $(".price").trigger("keyup");
            $(".discount").trigger("change");
            $("#tax_project").trigger('change');
            $(".tax_project_div").removeClass('d-none');

        });
    </script>
@endif

<script>
    $(document).on('click', '[data-repeater-create]', function() {
        $('.item :selected').each(function() {
            var id = $(this).val();
            if (id != '') {
                $(".item option[value=" + id + "]").addClass("d-none");
            }
        });
    })

    $(".tax_get").click(function() {
        myFunction();
    });
    $(".get_tax").change(function() {
        myFunction();
    });

    function myFunction() {

        var tax_id = $('.get_tax').val();

        if (tax_id != "") {
            $.ajax({
                url: "{{ route('get.taxes') }}",
                type: 'POST',
                headers: {
                    'X-CSRF-TOKEN': jQuery('#token').val()
                },
                data: {
                    'tax_id': tax_id,
                },
                cache: false,
                success: function(data) {
                    var obj = jQuery.parseJSON(data);


                    var taxes = '';
                    var tax = [];
                    $.each(obj, function() {

                        taxes += '<span class="badge bg-primary p-2 px-3 me-1 mr-1">' +
                            this.name + ' ' + '(' + this.rate + '%)' +
                            '</span>';
                        tax.push(this.id);

                    });

                    $('.taxes').html(taxes);
                },
            });
        } else {
            $('.taxes').html("");
        }
    }
</script>
