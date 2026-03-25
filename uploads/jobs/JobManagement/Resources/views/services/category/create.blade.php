<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('engineerings::modules.itemCategory.itemCategoryName')</h5>
    <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
</div>
<div class="modal-body">
    <x-table class="table-bordered" headType="thead-light">
        <x-slot name="thead">
            <th>#</th>
            <th>@lang('engineerings::modules.itemCategory.itemCategoryName')</th>
            <th class="text-right">@lang('app.action')</th>
        </x-slot>

        @forelse($categories as $key=>$category)
            <tr id="cat-{{ $category->id }}">
                <td>{{ $key + 1 }}</td>
                <td data-row-id="{{ $category->id }}" contenteditable="true">{{ mb_ucwords($category->name) }}
                </td>
                <td class="text-right">
                    <x-forms.button-secondary data-cat-id="{{ $category->id }}" icon="trash" class="delete-category">
                        @lang('app.delete')
                    </x-forms.button-secondary>
            </tr>
        @empty
            <x-cards.no-record-found-list colspan="3" />
        @endforelse
    </x-table>

    <x-form id="createProjectCategory">
        <div class="row border-top-grey ">
            <div class="col-sm-12">
                <x-forms.text fieldId="category_name" :fieldLabel="__('engineerings::modules.itemCategory.itemCategoryName')" fieldName="category_name" fieldRequired="true"
                    :fieldPlaceholder="__('placeholders.category')">
                </x-forms.text>
            </div>
        </div>
    </x-form>
</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.cancel')</x-forms.button-cancel>
    <x-forms.button-primary id="save-category" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    $('.delete-category').click(function() {

        const id = $(this).data('cat-id');
        let url = "{{ route('services-category.destroy', ':id') }}";
        url = url.replace(':id', id);

        const token = "{{ csrf_token() }}";

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
                $.easyAjax({
                    type: 'POST',
                    url: url,
                    data: {
                        '_token': token,
                        '_method': 'DELETE'
                    },
                    success: function(response) {
                        if (response.status === 'success') {
                            $('#cat-' + id).fadeOut();
                            $('#item_category_id').html(response.data);
                            $('#item_category_id').selectpicker('refresh');
                            // $(MODAL_LG).modal('hide');
                        }
                    }
                });
            }
        });

    });

    $('#save-category').click(function() {
        const url = "{{ route('services-category.store') }}";
        $.easyAjax({
            url: url,
            container: '#createProjectCategory',
            type: "POST",
            data: $('#createProjectCategory').serialize(),
            success: function(response) {
                if (response.status === 'success') {
                    $('#item_category_id').html(response.data);
                    $('#item_category_id').selectpicker('refresh');
                    $(MODAL_LG).modal('hide');
                }
            }
        })
    });

    $('[contenteditable=true]').focus(function() {
        $(this).data("initialText", $(this).html());
    }).blur(function() {
        if ($(this).data("initialText") !== $(this).html()) {
            let id = $(this).data('row-id');
            let value = $(this).html();

            let url = "{{ route('services-category.update', ':id') }}";
            url = url.replace(':id', id);

            const token = "{{ csrf_token() }}";

            $.easyAjax({
                url: url,
                container: '#row-' + id,
                type: "POST",
                data: {
                    'category_name': value,
                    '_token': token,
                    '_method': 'PUT'
                },
                blockUI: true,
                success: function(response) {
                    if (response.status == 'success') {
                        $('#item_category_id').html(response.data);
                        $('#item_category_id').selectpicker('refresh');
                    }
                }
            })
        }
    });
</script>
