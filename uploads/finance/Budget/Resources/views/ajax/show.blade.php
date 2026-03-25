@php
    if (!function_exists('format_currency')) {
        function format_currency($amount)
        {
            return number_format($amount, 2);
        }
    }
@endphp
<div class="row">
    <div class="col-sm-12">
        <x-cards.data :title="__('Budget Allocation Details')" class="mt-4">
            <x-slot name="action">
                <div class="dropdown">
                    <button class="btn f-14 px-0 py-0 text-dark-grey dropdown-toggle" type="button"
                        data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        <i class="fa fa-ellipsis-h"></i>
                    </button>

                    <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                        aria-labelledby="dropdownMenuLink" tabindex="0">
                        <a class="dropdown-item openRightModal"
                            href="{{ route('budgetallocationaprovalmodule.edit', [$allocation->id]) }}">
                            @lang('app.edit')
                        </a>
                        <a class="dropdown-item delete-table-row" href="javascript:;"
                            data-allocation-id="{{ $allocation->id }}">
                            @lang('app.delete')
                        </a>
                    </div>
                </div>
            </x-slot>

            <x-cards.data-row :label="__('Project Name')" :value="$allocation->project_name" />

            <x-cards.data-row :label="__('Department')" :value="$allocation->department" />

            <x-cards.data-row :label="__('Requested Budget')" :value="format_currency($allocation->budget_requested)" />

            <x-cards.data-row :label="__('Approved Budget')" :value="$allocation->budget_approved ? format_currency($allocation->budget_approved) : '--'" />

            <div class="col-12 px-0 pb-3 d-lg-flex d-md-flex d-block">
                <p class="mb-0 text-lightest f-14 w-30">@lang('Status')</p>
                <p class="mb-0 text-dark-grey f-14">
                    @php
                        $badgeColor = [
                            'pending' => 'warning',
                            'approved' => 'success',
                            'rejected' => 'danger'
                        ][$allocation->approval_status] ?? 'secondary';
                    @endphp
                    <span class="badge badge-{{ $badgeColor }}">@lang(ucfirst($allocation->approval_status))</span>
                </p>
            </div>

            <x-cards.data-row :label="__('Approval Date')" :value="$allocation->approval_date ? $allocation->approval_date->format(company()->date_format) : '--'" />

            <x-cards.data-row :label="__('Comments')" :value="$allocation->comments ?? '--'" html="true" />

        </x-cards.data>
    </div>
</div>

<script>
    $('body').on('click', '.delete-table-row', function () {
        var id = $(this).data('allocation-id');
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
                var url = "{{ route('budgetallocationaprovalmodule.destroy', ':id') }}";
                url = url.replace(':id', id);

                var token = "{{ csrf_token() }}";

                $.easyAjax({
                    type: 'POST',
                    url: url,
                    data: {
                        '_token': token,
                        '_method': 'DELETE'
                    },
                    success: function (response) {
                        if (response.status == "success") {
                            window.location.href = "{{ route('budgetallocationaprovalmodule.index') }}";
                        }
                    }
                });
            }
        });
    });
</script>