<div class="row mt-4">
    <div class="col-xl-7 col-lg-12 col-md-12 mb-4 mb-xl-0 mb-lg-4">
        <x-cards.data>
            <div class="row mb-3">
                <div class="col-10">
                    <h4 class="card-title f-15 f-w-500 text-darkest-grey mb-0">
                        @lang('engineerings::modules.meter') @lang('app.details')
                    </h4>
                </div>
                <div class="col-2 text-right">
                    <div class="dropdown">
                        <button class="btn f-14 px-0 py-0 text-dark-grey dropdown-toggle" type="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                            <i class="fa fa-ellipsis-h"></i>
                        </button>

                        <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                            aria-labelledby="dropdownMenuLink" tabindex="0">
                            <a class="dropdown-item" href="{{ route('meter.edit', $meter->id) }}">@lang('app.edit')</a>
                            <a class="dropdown-item delete-unit">@lang('app.delete')</a>
                        </div>
                    </div>
                </div>
            </div>
            <x-cards.data-row :label="__('engineerings::app.menu.unitID')" :value="$meter->unit->unit_name" />
            <x-cards.data-row :label="__('engineerings::app.menu.endMeter')" :value="mb_ucwords($meter->end_meter)" />
            <x-cards.data-row :label="__('engineerings::app.menu.typeBill')" :value="strtoupper($meter->type_bill)" />
            <x-cards.data-row :label="__('engineerings::app.menu.billingDate')" :value="$meter->billing_date" />
        </x-cards.data>
    </div>
    <div class="col-xl-5 col-lg-12 col-md-12 ">
        <x-cards.data>
            <div class="row mb-3">
                <div class="col-10">
                    <h4 class="card-title f-15 f-w-500 text-darkest-grey mb-0">
                        @lang('engineerings::modules.meter') @lang('engineerings::app.menu.image')
                    </h4>
                </div>
            </div>

            <div class="d-flex flex-wrap mt-3" id="task-file-list">
                <x-file-card :fileName="$meter->image_url" :dateAdded="$meter->created_at->diffForHumans()">
                    <img src="{{ $meter->image_url }}">

                    <x-slot name="action">
                        <div class="dropdown ml-auto file-action">
                            <button class="btn btn-lg f-14 p-0 text-lightest text-capitalize rounded  dropdown-toggle"
                                type="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                <i class="fa fa-ellipsis-h"></i>
                            </button>

                            <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                                aria-labelledby="dropdownMenuLink" tabindex="0">
                                <a class="cursor-pointer d-block text-dark-grey f-13 p-3" target="_blank"
                                    href="{{ $meter->image_url }}">@lang('app.view')</a>
                            </div>
                        </div>
                    </x-slot>
                </x-file-card>
            </div>
        </x-cards.data>
    </div>
</div>
<!-- ROW END -->

<script>
    $('body').on('click', '.delete-table-row', function() {
            var id = $(this).data('unit-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('meter.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                showTable();
                            }
                        }
                    });
                }
            });
        });
</script>
