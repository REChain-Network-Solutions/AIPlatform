<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('modules.tickets.manageGroups')</h5>
    <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span
            aria-hidden="true">×</span>
    </button>
</div>

<div class="modal-body">
    <div class="portlet-body">
        <div class="table-responsive">
            <x-table class="table-bordered">
                <x-slot name="thead">
                    <th>#</th>
                    <th>@lang('modules.tickets.group')</th>
                    <th class="text-right">@lang('app.action')</th>
                </x-slot>

                @forelse($groups as $key=>$group)
                    <tr id="group-{{ $group->id }}">
                        <td>{{ $key+1 }}</td>
                        <td>{{ mb_ucwords($group->group_name) }}</td>
                        <td class="text-right">
                            <div class="task_view">
                                <a href="javascript:;" class="delete-group task_view_more d-flex align-items-center justify-content-center" data-group-id="{{ $group->id }}">
                                    <i class="fa fa-trash icons mr-2"></i> @lang('app.delete')
                                </a>
                            </div>
                        </td>
                    </tr>
                @empty
                    <tr>
                        <td colspan="3">@lang('messages.noGroupAdded')</td>
                    </tr>
                @endforelse
            </x-table>

        </div>

        <hr>
        <x-form id="createTicketGroup" method="POST" class="ajax-form">
            <div class="form-body">
                <div class="row">
                    <div class="col-lg-12">
                        <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.tickets.groupName')"
                            fieldPlaceholder="e.g. Space X" fieldRequired="true" fieldName="group_name"
                            fieldId="group_name"/>
                    </div>
                </div>
            </div>
        </x-form>
    </div>
</div>

<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.cancel')</x-forms.button-cancel>
    <x-forms.button-primary id="save-group" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    $('#createTicketGroup').on('submit', function(e) {
        return false;
    })

    $('.delete-group').click(function () {

        var id = $(this).data('group-id');

        Swal.fire({
            title: "@lang('messages.sweetAlertTitle')",
            text: "@lang('messages.removeGroupText')",
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
            if (result.isConfirmed)
            {

                var url = "{{ route('complaint-groups.destroy',':id') }}";
                url = url.replace(':id', id);

                var token = "{{ csrf_token() }}";

                $.easyAjax({
                    type: 'POST',
                    url: url,
                    blockUI: true,
                    data: {'_token': token, '_method': 'DELETE'},
                    success: function (response) {
                        if (response.status == "success") {
                            window.location.reload();
                        }
                    }
                });
            }
        });
    });

    $('#save-group').click(function () {
        $.easyAjax({
            url: "{{route('complaint-groups.store')}}",
            container: '#createTicketGroup',
            type: "POST",
            blockUI: true,
            data: $('#createTicketGroup').serialize(),
            success: function (response) {
                if(response.status == 'success'){
                    $('#group_id').html(response.data);
                    $('#group_id').selectpicker('refresh');
                    $("[data-dismiss=modal]").trigger({ type: "click" });
                }
            }
        })
    });
</script>
