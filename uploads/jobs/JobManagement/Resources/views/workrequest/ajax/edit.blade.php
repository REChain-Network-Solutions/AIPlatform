@extends('layouts.app')

@section('content')
    <div class="content-wrapper">
        <div class="bg-white rounded b-shadow-4 create-inv">
            <!-- HEADING START -->
            <div class="px-lg-4 px-md-4 px-3 py-3">
                <h4 class="mb-0 f-21 font-weight-normal text-capitalize">@lang('engineerings::modules.wr') @lang('app.details')</h4>
            </div>
            <!-- HEADING END -->
            <hr class="m-0 border-top-grey">
            <!-- FORM START -->
            <x-form class="c-inv-form" id="saveInvoiceForm"> @method('PUT')
                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <div class="col-md-2">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label class="mb-12" fieldId="wr_no" :fieldLabel="__('engineerings::app.menu.noWR')" fieldRequired="true">
                            </x-forms.label>
                            <x-forms.input-group>
                                <input type="text" name="wr_no" id="wr_no" class="form-control height-35 f-15"
                                    value="{{ $wr->wr_no }}">
                            </x-forms.input-group>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="complaint_id" :fieldLabel="__('engineerings::app.menu.ticketID')" fieldName="complaint_id"
                                fieldRequired="true">
                            </x-forms.label>
                            <x-forms.input-group>
                                <select class="form-control select-picker" name="complaint_id" id="complaint_id">
                                    @foreach ($ticket as $items)
                                        <option @if ($items->id == $wr->complaint_id) selected @endif
                                            value="{{ $items->id }}">{{ ucwords($items->subject) }}</option>
                                    @endforeach
                                </select>
                            </x-forms.input-group>
                        </div>
                    </div>
                    <div class="col-lg-2">
                        <x-forms.label fieldId="check_time" :fieldLabel="__('engineerings::app.menu.checkTime')" fieldRequired="true"></x-forms.label>
                        <div class="bootstrap-timepicker timepicker">
                            <input type="datetime-local" id="check_time" name="check_time"
                                class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                                value="{{ $wr->check_time }}">
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="user_id" :fieldLabel="__('engineerings::app.menu.assignTo')" fieldName="user_id" fieldRequired="true">
                            </x-forms.label>
                            <x-forms.input-group>
                                <select class="form-control select-picker" name="user_id" id="user_id"
                                    data-live-search="true">
                                    @foreach ($employees as $items)
                                        <option @if ($items->id == $wr->assign_to) selected @endif
                                            value="{{ $items->id }}">{{ $items->name }}</option>
                                    @endforeach
                                </select>
                            </x-forms.input-group>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <label class="f-14 text-dark-grey mb-12 w-100" for="usr">@lang('engineerings::app.menu.tenant')</label>
                            <div class="d-flex">
                                <x-forms.radio fieldId="notification-yes" :fieldLabel="__('app.yes')" fieldValue="yes"
                                    fieldName="charge_by_tenant" checked="true">
                                </x-forms.radio>
                                <x-forms.radio fieldId="notification-no" :fieldLabel="__('app.no')" fieldValue="no"
                                    fieldName="charge_by_tenant">
                                </x-forms.radio>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-12">
                        <div class="form-group">
                            <x-forms.label class="my-3" fieldId="description-text" :fieldLabel="__('engineerings::app.menu.remark')" fieldRequired="true">
                            </x-forms.label>
                            <textarea name="remark" id="description-text" rows="5" class="form-control">{{ $wr->remark }}</textarea>
                        </div>
                    </div>
                    <div class="col-md-12">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label class="mb-12" fieldId="problem" :fieldLabel="__('engineerings::app.menu.problem')" fieldRequired="true">
                            </x-forms.label>
                            <x-forms.input-group>
                                <input type="text" name="problem" id="problem" class="form-control height-35 f-15"
                                    value="{{ $wr->problem }}">
                            </x-forms.input-group>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('complaint::app.menu.area')" fieldName="parent_label">
                        </x-forms.label>
                        <select class="form-control select-picker" name="area_id" id="area_id" data-live-search="true">
                            <option value="">No Items Selected</option>
                            @foreach ($areas as $area)
                                <option @if ($area->id == $wr->house->area->id) selected @endif value="{{ $area->id }}">
                                    {{ $area->area_name }}</option>
                            @endforeach
                        </select>
                    </div>
                    <div class="col-md-6">
                        <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('complaint::app.menu.houses')" fieldName="parent_label">
                        </x-forms.label>
                        <select class="form-control select-picker" name="house_id" id="house_id" data-row-id="0"
                            data-live-search="true">
                            <option value="">No Items Selected</option>
                            <option value="{{ $wr->house->house_name }}" selected>{{ $wr->house->house_name }}</option>
                        </select>
                    </div>
                    <div class="col-lg-12">
                        <x-forms.file allowedFileExtensions="png jpg jpeg svg" class="mr-0 mr-lg-2 mr-md-2 cropper"
                            :fieldLabel="__('engineerings::app.menu.foto')" fieldName="image" fieldId="image" fieldHeight="119" :fieldValue="$url" />
                    </div>
                </div>
                <!-- CLIENT, PROJECT, GST, BILLING, SHIPPING ADDRESS END -->
                <hr class="mb-4 border-top-grey">
                <!--  ADD ITEM START-->
                <div class="row px-lg-4 px-md-4 px-3 pt-0 mt-3 mb-3">
                    <div class="col-md-12">
                        <a class="f-15 f-w-500 mr-4" href="javascript:;" id="add-item">
                            <i class="icons icon-plus font-weight-bold mr-1"></i>@lang('modules.invoices.addItem')
                        </a>
                        <a class="f-15 f-w-500" href="javascript:;" id="add-service">
                            <i class="icons icon-plus font-weight-bold mr-1"></i>@lang('modules.invoices.addService')
                        </a>
                    </div>
                </div>

                <div id="sortable">
                    @if (isset($wr))
                        <!-- DESKTOP DESCRIPTION TABLE START -->
                        <div class="d-flex px-4 c-inv-desc item-row">
                            <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                                <table width="100%">
                                    <tbody>
                                        @foreach ($wr->items as $key => $item)
                                            <tr class="text-dark-grey font-weight-bold f-14">
                                                <td width="35%" class="border">@lang('Item')</td>
                                                <td width="12%" class="border" align="right">@lang('engineerings::app.menu.qty')</td>
                                                <td width="16%" class="border" align="right">@lang('engineerings::app.menu.harga')</td>
                                                <td width="17%" class="border" align="right">@lang('engineerings::app.menu.tax')</td>
                                                <td width="18%" class="border" align="right">@lang('engineerings::app.menu.total')</td>
                                                <td width="2%" class="border-0" align="right"></td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div class="select-others height-35 rounded border-0">
                                                        <input type="hidden" name="items_ids[]"
                                                            value="{{ $item->id }}">
                                                        <select class="form-control select-picker" name="items_id[]">
                                                            @foreach ($items_arr as $item_arr)
                                                                <option @if ($item_arr->id == $item->items_id) selected @endif
                                                                    value="{{ $item_arr->id }}">
                                                                    {{ ucwords($item_arr->name) }}
                                                                </option>
                                                            @endforeach
                                                        </select>
                                                    </div>
                                                </td>
                                                <td>
                                                    <input type="number" min="1"
                                                        class="form-control f-14 height-35 rounded border-0 w-100 text-right item_name"
                                                        name="items_qty[]" value="{{ $item->qty }}">
                                                </td>
                                                <td>
                                                    <input type="number" min="100"
                                                        class="form-control f-14 height-35 rounded w-100 text-right"
                                                        name="items_harga[]" id="harga" value="{{ $item->harga }}">
                                                </td>
                                                <td>
                                                    <div class="select-others height-35 rounded border-0">
                                                        <select name="items_tax[]" class="form-control select-picker">
                                                            <option value="0">--</option>
                                                            @foreach ($taxes as $tax)
                                                                <option data-rate="{{ $tax->rate_percent }}"
                                                                    @if ($tax->rate_percent == $item->tax) selected @endif
                                                                    value="{{ $tax->rate_percent }}">
                                                                    {{ $tax->rate_percent }}%
                                                                </option>
                                                            @endforeach
                                                        </select>
                                                    </div>
                                                </td>
                                                <td align="right">
                                                    <span class="jumlah">0.00</span>
                                                </td>
                                                <td class="border-0">
                                                    <a href="javascript:;"
                                                        class="d-flex align-items-center justify-content-center remove-item"><i
                                                            class="fa fa-times-circle f-20 text-lightest"></i></a>
                                                </td>
                                            </tr>
                                        @endforeach
                                        @foreach ($wr->services as $key => $service)
                                            <tr class="text-dark-grey font-weight-bold f-14">
                                                <td width="35%" class="border">@lang('Service')</td>
                                                <td width="12%" class="border" align="right">@lang('engineerings::app.menu.qty')</td>
                                                <td width="16%" class="border" align="right">@lang('engineerings::app.menu.harga')</td>
                                                <td width="17%" class="border" align="right">@lang('engineerings::app.menu.tax')</td>
                                                <td width="18%" class="border" align="right">@lang('engineerings::app.menu.total')</td>
                                                <td width="2%" class="border-0" align="right"></td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div class="select-others height-35 rounded border-0">
                                                        <input type="hidden" name="services_ids[]"
                                                            value="{{ $service->id }}">
                                                        <select class="form-control select-picker" name="services_id[]">
                                                            @foreach ($services as $item)
                                                                <option @if ($item->id == $service->items_id) selected @endif
                                                                    value="{{ $item->id }}">
                                                                    {{ ucwords($item->name) }}
                                                                </option>
                                                            @endforeach
                                                        </select>
                                                    </div>
                                                </td>
                                                <td>
                                                    <input type="number" min="1"
                                                        class="form-control f-14 height-35 rounded border-0 w-100 text-right item_name"
                                                        name="services_qty[]" value="{{ $service->qty }}">
                                                </td>
                                                <td>
                                                    <input type="number" min="100"
                                                        class="form-control f-14 height-35 rounded w-100 text-right"
                                                        name="services_harga[]" id="harga"
                                                        value="{{ $service->harga }}">
                                                </td>
                                                <td>
                                                    <div class="select-others height-35 rounded border-0">
                                                        <select name="services_tax[]" class="form-control select-picker">
                                                            <option value="0">--</option>
                                                            @foreach ($taxes as $tax)
                                                                <option data-rate="{{ $tax->rate_percent }}"
                                                                    @if ($tax->rate_percent == $service->tax) selected @endif
                                                                    value="{{ $tax->rate_percent }}">
                                                                    {{ $tax->rate_percent }}%
                                                                </option>
                                                            @endforeach
                                                        </select>
                                                    </div>
                                                </td>
                                                <td align="right">
                                                    <span class="jumlah">0.00</span>
                                                </td>
                                                <td class="border-0">
                                                    <a href="javascript:;"
                                                        class="d-flex align-items-center justify-content-center remove-item"><i
                                                            class="fa fa-times-circle f-20 text-lightest"></i></a>
                                                </td>
                                            </tr>
                                        @endforeach
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <!-- DESKTOP DESCRIPTION TABLE END -->
                    @else
                        <!-- DESKTOP DESCRIPTION TABLE START -->
                        <div class="d-flex px-4 c-inv-desc item-row">
                            <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                                <table width="100%">
                                    <tbody>
                                        <tr class="text-dark-grey font-weight-bold f-14">
                                            <td width="35%" class="border">@lang('Items')</td>
                                            <td width="12%" class="border" align="right">@lang('Qty')</td>
                                            <td width="16%" class="border" align="right">@lang('Harga')</td>
                                            <td width="17%" class="border" align="right">@lang('Tax')</td>
                                            <td width="18%" class="border" align="right">@lang('Total /items')</td>
                                            <td width="2%" class="border-0" align="right"></td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <div class="select-others height-35 rounded border-0">
                                                    <select class="form-control select-picker" name="items_id[]">
                                                        @foreach ($items_arr as $item)
                                                            <option value="{{ $item->id }}">
                                                                {{ ucwords($item->name) }}
                                                            </option>
                                                        @endforeach
                                                    </select>
                                                </div>
                                            </td>
                                            <td>
                                                <input type="number" min="1"
                                                    class="form-control f-14 height-35 rounded border-0 w-100 text-right item_name"
                                                    name="item_name[]">
                                            </td>
                                            <td>
                                                <input type="number" min="100"
                                                    class="form-control f-14 height-35 rounded w-100 text-right"
                                                    name="harga[]" id="harga">
                                            </td>
                                            <td>
                                                <div class="select-others height-35 rounded border-0">
                                                    <select name="tax[]" class="form-control select-picker">
                                                        <option value="0">--</option>
                                                        @foreach ($taxes as $tax)
                                                            <option data-rate="{{ $tax->rate_percent }}"
                                                                value="{{ $tax->rate_percent }}">
                                                                {{ $tax->rate_percent }}%
                                                            </option>
                                                        @endforeach
                                                    </select>
                                                </div>
                                            </td>
                                            <td align="right">
                                                <span class="jumlah">0.00</span>
                                            </td>
                                            <td class="border-0">
                                                <a href="javascript:;"
                                                    class="d-flex align-items-center justify-content-center remove-item"><i
                                                        class="fa fa-times-circle f-20 text-lightest"></i></a>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <!-- DESKTOP DESCRIPTION TABLE END -->
                    @endif
                </div>

                <div class="d-flex px-4 mb-2 c-inv-desc item-row">
                    <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                        <table width="100%">
                            <tbody>
                                <tr class="text-dark-grey font-weight-bold f-14">
                                    <td width="35%" class="border-0"></td>
                                    <td width="12%" class="border-0"></td>
                                    <td width="16%" class="border-0"></td>
                                    <td width="17%" class="border-0"></td>
                                    <td width="18%" class="border-0"></td>
                                    <td width="2%" class="border-0"></td>
                                </tr>
                                <tr class="d-none d-md-table-row d-lg-table-row f-14">
                                    <td colspan="4" class="dash-border-top border bg-amt-grey" align="right">
                                        <b>@lang('modules.invoices.total')</b>
                                    </td>
                                    <td class="dash-border-top border"><span class="jumlah-total">0.00</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- CANCEL SAVE START -->
                <x-form-actions class="c-inv-btns d-block d-lg-flex d-md-flex">
                    <x-forms.button-primary class="save-form mr-3" icon="check">@lang('app.save')
                    </x-forms.button-primary>

                    <x-forms.button-cancel :link="route('engineerings.index')" class="border-0 ">@lang('app.cancel')
                    </x-forms.button-cancel>
                </x-form-actions>
                <!-- CANCEL SAVE END -->

            </x-form>
            <!-- FORM END -->
        </div>
    </div>
@endsection

@push('scripts')
    <script>
        $(document).ready(function() {
            $(document).on('change', 'select[name="area_id"]', function() {
                var selectedId = $(this).val();
                houseItems(selectedId);
            });

            function houseItems(selectedId) {
                $('.select-picker').selectpicker();
                $.ajax({
                    url: '{{ route('complaint.get-items', '') }}' + '/' + selectedId,
                    method: 'GET',
                    success: function(data) {
                        $('select[name^="house_id"]').each(function(index, select) {
                            var $select = $(select);
                            var selectedValue = $select.val();
                            var parentRow = $(this).closest('tr');
                            var qtyInput = parentRow.find('input[name="house_id"]');
                            $select.empty();
                            $.each(data, function(idx, item) {
                                var option = $('<option>', {
                                    value: item.id,
                                    text: item.house_name
                                });
                                $select.append(option);
                            });
                            $select.val(selectedValue);
                            $select.selectpicker('refresh');
                            qtyInput.val('');
                        });
                    }
                });
            }

            $('input[name^="items_qty"]').each(function() {
                hitungItem($(this).closest('tr'));
            });

            $('input[name^="services_qty"]').each(function() {
                hitungService($(this).closest('tr'));
            });

            hitungTotalSemua();

            $(document).on('click', '#add-item', function() {
                var i = $(document).find('.item_name').length;
                var item =
                    ` <div class="d-flex px-4 c-inv-desc item-row">
                <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                <table width="100%">
                <tbody>
                    <tr class="text-dark-grey font-weight-bold f-14">
                        <td width="35%" class="border">@lang('Items')</td>
                        <td width="12%" class="border" align="right">@lang('engineerings::app.menu.qty')</td>
                        <td width="16%" class="border" align="right">@lang('engineerings::app.menu.harga')</td>
                        <td width="17%" class="border" align="right">@lang('engineerings::app.menu.tax')</td>
                        <td width="18%" class="border" align="right">@lang('engineerings::app.menu.total')</td>
                        <td width="2%" class="border-0" align="right"></td>
                    </tr>
                    <tr>
                        <td width="35%">
                            <div class="select-others height-35 rounded border-0">
                                <select class="form-control select-picker height-35 f-14" name="items_id[]">
                                    <option value="">--</option>
                                    @foreach ($items_arr as $item)
                                        <option value="{{ $item->id }}">{{ $item->name }}</option>
                                    @endforeach
                                </select>
                            </div>
                        </td>
                        <td width="12%">
                            <input type="number" min="1"
                                class="form-control f-14 height-35 rounded border-0 w-100 text-right item_name" name="items_qty[]">
                        </td>
                        <td width="16%">
                            <input type="number" min="100"
                                class="form-control f-14 height-35 rounded w-100 text-right" name="items_harga[]" id="harga" readonly>
                        </td>
                        <td width="17%">
                            <div class="select-others height-35 rounded border-0">
                                <select class="form-control select-picker height-35 f-14" name="items_tax[]">
                                    <option value="0">--</option>
                                    @foreach ($taxes as $tax)
                                        <option data-rate="{{ $tax->rate_percent }}"
                                            value="{{ $tax->rate_percent }}">{{ strtoupper($tax->tax_name) }}:
                                            {{ $tax->rate_percent }}%</option>
                                    @endforeach
                                </select>
                            </div>
                        </td>
                        <td width="18%" align="right">
                            <span class="jumlah">0.00</span>
                        </td>
                        <td class="border-0">
                            <a href="javascript:;"
                                class="d-flex align-items-center justify-content-center remove-item"><i
                                    class="fa fa-times-circle f-20 text-lightest"></i></a>
                        </td>
                    </tr>
                </tbody>
                </table>
                </div>
                </div>`;
                $(item).hide().appendTo("#sortable").fadeIn(500);
                $('.select-picker').selectpicker();
            });

            $(document).on('click', '#add-service', function() {
                var i = $(document).find('.item_name').length;
                var item =
                    ` <div class="d-flex px-4 c-inv-desc item-row">
                <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                <table width="100%">
                <tbody>
                    <tr class="text-dark-grey font-weight-bold f-14">
                        <td width="35%" class="border">@lang('Services')</td>
                        <td width="12%" class="border" align="right">@lang('engineerings::app.menu.qty')</td>
                        <td width="16%" class="border" align="right">@lang('engineerings::app.menu.harga')</td>
                        <td width="17%" class="border" align="right">@lang('engineerings::app.menu.tax')</td>
                        <td width="18%" class="border" align="right">@lang('engineerings::app.menu.total')</td>
                        <td width="2%" class="border-0" align="right"></td>
                    </tr>
                    <tr>
                        <td width="35%">
                            <div class="select-others height-35 rounded border-0">
                                <x-forms.input-group>
                                    <select class="form-control select-picker height-35 f-14 services_id" id="services_id" name="services_id[]">
                                        <option value="">--</option>
                                        @foreach ($services as $service)
                                            <option value="{{ $service->id }}">{{ $service->name }}</option>
                                        @endforeach
                                    </select>
                                    <x-slot name="append">
                                        <a href="{{ route('services.index') }}" class="btn btn-outline-secondary border-grey openRightModal float-left">
                                            @lang('app.add')
                                        </a>
                                    </x-slot>
                                </x-forms.input-group>
                            </div>
                        </td>
                        <td width="12%">
                            <input type="number" min="1"
                                class="form-control f-14 height-35 rounded border-0 w-100 text-right item_name" name="services_qty[]">
                        </td>
                        <td width="16%">
                            <input type="number" min="100"
                                class="form-control f-14 height-35 rounded w-100 text-right" name="services_harga[]" id="harga" readonly>
                        </td>
                        <td width="17%">
                            <div class="select-others height-35 rounded border-0">
                                <select class="form-control select-picker height-35 f-14" name="services_tax[]">
                                    <option value="0">--</option>
                                    @foreach ($taxes as $tax)
                                        <option data-rate="{{ $tax->rate_percent }}"
                                            value="{{ $tax->rate_percent }}">{{ strtoupper($tax->tax_name) }}:
                                            {{ $tax->rate_percent }}%</option>
                                    @endforeach
                                </select>
                            </div>
                        </td>
                        <td width="18%" align="right">
                            <span class="jumlah">0.00</span>
                        </td>
                        <td class="border-0">
                            <a href="javascript:;"
                                class="d-flex align-items-center justify-content-center remove-item"><i
                                    class="fa fa-times-circle f-20 text-lightest"></i></a>
                        </td>
                    </tr>
                </tbody>
                </table>
                </div>
                </div>`;
                $(item).hide().appendTo("#sortable").fadeIn(500);
                $('.select-picker').selectpicker();
            });

            $('#saveInvoiceForm').on('click', '.remove-item', function() {
                $(this).closest('.item-row').fadeOut(300, function() {
                    $(this).remove();
                });
            });

            $(document).on('keyup', 'input[name^="items_qty"]', function() {
                hitungItem($(this).closest('tr'));
                hitungTotalSemua();
            });

            $(document).on('keyup', 'input[name^="services_qty"]', function() {
                hitungService($(this).closest('tr'));
                hitungTotalSemua();
            });

            $(document).on('change', 'select[name^="items_tax"], select[name^="services_tax"]', function() {
                hitungTotalBaris($(this).closest('tr'));
                hitungTotalSemua();
            });

            $(document).on('change', 'select[name^="items_id"]', function() {
                var itemID = $(this).val();
                var parentRow = $(this).closest('tr');
                getItems(itemID, parentRow);
            });

            function getItems(itemID, parentRow) {
                $.ajax({
                    url: '{{ route('engineerings.get_items', '') }}' + '/' + itemID,
                    method: 'GET',
                    success: function(data) {
                        if (Array.isArray(data) && data.length > 0) {
                            var harga = parseFloat(data[0].price);
                            var hargaInput = parentRow.find('input[name="items_harga[]"]');
                            hargaInput.val(harga);
                        }
                    }
                });
            }

            $(document).on('change', 'select[name^="services_id"]', function() {
                var serviceID = $(this).val();
                var parentRow = $(this).closest('tr');
                getService(serviceID, parentRow);
            });

            function getService(serviceID, parentRow) {
                $.ajax({
                    url: '{{ route('engineerings.get_services', '') }}' + '/' + serviceID,
                    method: 'GET',
                    success: function(data) {
                        if (Array.isArray(data) && data.length > 0) {
                            var harga = parseFloat(data[0].price);
                            var hargaInput = parentRow.find('input[name="services_harga[]"]');
                            hargaInput.val(harga);
                        }
                    }
                });
            }

            function hitungItem(row) {
                const qtyItem = Number(row.find('input[name^="items_qty"]').val()) || 0;
                const priceItem = Number(row.find('input[name^="items_harga"]').val());
                const taxItem = Number(row.find('select[name^="items_tax"]').val());
                const taxRateItem = Number(row.find('select[name^="items_tax"] option:selected').data('rate')) ||
                    0;

                const totalItem = qtyItem * priceItem;

                let jml_tax_item = 0;
                let jml_total = 0;

                if (taxItem !== 0) {
                    jml_tax_item = (taxRateItem / 100) * totalItem;
                }

                jml_total = jml_tax_item + totalItem;
                row.find('span.jumlah').text(jml_total);
            }

            function hitungService(row) {
                const qtyService = Number(row.find('input[name^="services_qty"]').val()) || 0;
                const priceService = Number(row.find('input[name^="services_harga"]').val());
                const taxService = Number(row.find('select[name^="services_tax"]').val());
                const taxRateService = Number(row.find('select[name^="services_tax"] option:selected').data(
                        'rate')) ||
                    0;

                const totalService = qtyService * priceService;

                let jml_tax_service = 0;
                let jml_total = 0;

                if (taxService !== 0) {
                    jml_tax_service = (taxRateService / 100) * totalService;
                }

                jml_total = jml_tax_service + totalService;
                row.find('span.jumlah').text(jml_total);
            }

            function hitungTotalSemua() {
                const semuaJumlah = Array.from(document.querySelectorAll('span.jumlah'));
                const total = semuaJumlah.map(jumlah => parseInt(jumlah.innerHTML.replace(/\D/g, ''))).reduce((
                    total, jumlah) => total + jumlah, 0);

                document.querySelector('span.jumlah-total').textContent = total;
            }

            $('.save-form').click(function() {
                var type = $(this).data('type');
                $('#type').val(type);

                $.easyAjax({
                    url: "{{ route('engineerings.update', $wr->id) }}",
                    container: '#saveInvoiceForm',
                    type: "POST",
                    blockUI: true,
                    redirect: true,
                    file: true,
                    data: $('#saveInvoiceForm').serialize(),
                    success: function(response) {
                        if (response.status === 'success') {
                            window.location.href = response.redirectUrl;
                        }
                    }
                });
            });

            init(RIGHT_MODAL);
        });
    </script>
@endpush
