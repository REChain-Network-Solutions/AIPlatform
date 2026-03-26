@php
    $addSchedulePermission = user()->permission('add_quality_control');
@endphp

<!-- ROW START -->
<div class="row py-5">
    <div class="col-lg-12 col-md-12 mb-4 mb-xl-0 mb-lg-4">
         <!-- Add Task Export Buttons Start -->
        <div class="d-flex" id="table-actions">
        </div>
        <!-- Add Task Export Buttons End -->
        <!-- Task Box Start -->
        <div class="d-flex flex-column w-tables rounded mt-3 bg-white">
            {!! $dataTable->table(['class' => 'table table-hover border-0 w-100']) !!}
        </div>
        <!-- Task Box End -->
    </div>
</div>
<!-- ROW END -->
@include('sections.datatable_js')

<script>
    $('#recurring-inspection_schedules-table').on('preXhr.dt', function(e, settings, data) {

        var projectID = "{{ $schedule->project_id }}";
        var recurringID = "{{ $recurringID }}";
        data['projectID'] = projectID;
        data['recurringID'] = recurringID;
    });
    const showTable = () => {
        window.LaravelDataTables["recurring-inspection_schedules-table"].draw(false);
    }

    $('body').on('click', '.cancel-schedule', function() {
        var id = $(this).data('schedule-id');
        Swal.fire({
            title: "@lang('messages.sweetAlertTitle')",
            text: "@lang('messages.scheduleText')",
            icon: 'warning',
            showCancelButton: true,
            focusConfirm: false,
            confirmButtonText: "@lang('app.yes')",
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

                var url = "{{ route('inspection_schedules.update_status',':id') }}";
                url = url.replace(':id', id);

                var token = "{{ csrf_token() }}";

                $.easyAjax({
                    type: 'GET',
                    url: url,
                    success: function(response) {
                        if (response.status == "success") {
                            window.LaravelDataTables["recurring-inspection_schedules-table"].draw(false);
                        }
                    }
                });
            }
        });
    });










    $('body').on('click', '#recurring-schedule', function() {
        window.location.href = "{{ route('recurring-inspection_schedules.index')}} ";
    });





</script>
