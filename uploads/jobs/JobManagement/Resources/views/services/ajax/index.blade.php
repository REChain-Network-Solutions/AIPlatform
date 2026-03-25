<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">

<div class="d-flex justify-content-between action-bar">
    <div id="table-actions" class="flex-grow-1 align-items-center">
        <x-forms.link-primary :link="route('services.create')" class="mr-3 openRightModal float-left" icon="plus">
            @lang('modules.invoices.addService')
        </x-forms.link-primary>
    </div>
</div>
<div class="d-flex flex-column w-tables rounded mt-3 bg-white">
    <x-table class="table-bordered" headType="thead-light">
        <x-slot name="thead">
            <th>#</th>
            <th>@lang('app.image')</th>
            <th>@lang('app.name')</th>
            <th>@lang('engineerings::modules.itemCategory.itemCategory')</th>
            <th>@lang('engineerings::modules.itemCategory.itemSubCategory')</th>
            <th>@lang('app.description')</th>
            <th>@lang('app.price')</th>
            <th>@lang('app.action')</th>
        </x-slot>

        @forelse($services as $key => $service)
            <tr id="cat-{{ $service->id }}">
                <td>{{ $key + 1 }}</td>
                <td>
                    @if ($service->image_url == null)
                        no image
                    @else
                        <img src="{{ $service->image_url }}" class="border rounded height-35" />
                    @endif
                </td>
                <td>{{ $service->name }}</td>
                <td>{{ $service->category->name ?? '--' }}</td>
                <td>{{ $service->subCategory->name ?? '--' }}</td>
                <td>{{ $service->description ?? '--' }}</td>
                <td>{{ $service->price ?? '--' }}</td>
                <td>
                    <div class="action-bar">
                        <x-forms.link-primary :link="route('services.edit', [$service->id])" class="mr-3 openRightModal float-left" icon="edit">
                            @lang('app.edit')
                        </x-forms.link-primary>
                        <x-forms.button-secondary data-cat-id="{{ $service->id }}" icon="trash"
                            class="delete-service">
                            @lang('app.delete')
                        </x-forms.button-secondary>
                    </div>
                </td>
            </tr>
        @empty
            <x-cards.no-record-found-list colspan="8" />
        @endforelse
    </x-table>
</div>

<script>
    $('.delete-service').click(function() {
        const id = $(this).data('cat-id');
        let url = "{{ route('services.destroy', ':id') }}";
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
                            var responseOptions = response.data;
                            $('select[name="services_id[]"]').append(responseOptions);
                            $('select[name="services_id[]"]').selectpicker('refresh');
                        }
                    }
                });
            }
        });

    });
</script>
