<div class="row">
    <div class="col-sm-12">
        <x-cards.data :title="__('app.menu.tasksAssignment') . ' ' . __('app.details')" class="mt-4">
            <x-slot name="action">
                <div class="dropdown">
                    <button class="btn f-14 px-0 py-0 text-dark-grey dropdown-toggle" type="button"
                        data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        <i class="fa fa-ellipsis-h"></i>
                    </button>

                    <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                        aria-labelledby="dropdownMenuLink" tabindex="0">
                        <a class="dropdown-item openRightModal" href="{{ route('tasksassignmentmodule.edit', [$task->id]) }}">
                            @lang('app.edit')
                        </a>
                        <a class="dropdown-item delete-table-row" href="javascript:;" data-task-id="{{ $task->id }}">
                            @lang('app.delete')
                        </a>
                    </div>
                </div>
            </x-slot>

            <x-cards.data-row :label="__('Date')" 
                :value="$task->date ? $task->date->format(company()->date_format) : '--'" />

            <x-cards.data-row :label="__('Task')" :value="$task->task" />

            <x-cards.data-row :label="__('Assigned To')" :value="$task->assign_to" />

            <x-cards.data-row :label="__('Product')" :value="$task->product" />

            <div class="col-12 px-0 pb-3 d-lg-flex d-md-flex d-block">
                <p class="mb-0 text-lightest f-14 w-30">@lang('Priority')</p>
                <p class="mb-0 text-dark-grey f-14">
                    @php
                        $badgeColor = [
                            'High' => 'danger',
                            'Medium' => 'warning',
                            'Low' => 'primary'
                        ][$task->priority] ?? 'secondary';
                    @endphp
                    <span class="badge badge-{{ $badgeColor }}">{{ $task->priority }}</span>
                </p>
            </div>

            <div class="col-12 px-0 pb-3 d-lg-flex d-md-flex d-block">
                <p class="mb-0 text-lightest f-14 w-30">@lang('Status')</p>
                <p class="mb-0 text-dark-grey f-14">
                    @php
                        $badgeColor = [
                            'Not Yet Started' => 'secondary',
                            'In Progress' => 'info',
                            'Completed' => 'success'
                        ][$task->status] ?? 'secondary';
                    @endphp
                    <span class="badge badge-{{ $badgeColor }}">{{ $task->status }}</span>
                </p>
            </div>

            <x-cards.data-row :label="__('ETA')" :value="$task->eta ?? '--'" />

            <x-forms.custom-field-show :fields="$fields" :model="$task"></x-forms.custom-field-show>

        </x-cards.data>
    </div>
</div>

<script>
    $('body').on('click', '.delete-table-row', function() {
        var id = $(this).data('task-id');
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
                var url = "{{ route('tasksassignmentmodule.destroy', ':id') }}";
                url = url.replace(':id', id);

                var token = "{{ csrf_token() }}";

                $.easyAjax({
                    type: 'POST',
                    url: url,
                    data: {
                        '_token': token,
                        '_method': 'DELETE'
                    },
                    success: function(response) {
                        if (response.status == "success") {
                            window.location.href = "{{ route('tasksassignmentmodule.index') }}";
                        }
                    }
                });
            }
        });
    });
</script>