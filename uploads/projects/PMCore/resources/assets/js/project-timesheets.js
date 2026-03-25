$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // DataTable
    let timesheetsTable = $('#timesheetsTable').DataTable({
        processing: true,
        serverSide: true,
        ajax: {
            url: pageData.urls.datatable,
            data: function (d) {
                d.project_id = pageData.projectId;
            }
        },
        columns: [
            { data: 'formatted_date', name: 'date' },
            { data: 'user', name: 'user.name', orderable: false },
            { data: 'task_name', name: 'task.title' },
            { data: 'description', name: 'description' },
            { data: 'formatted_hours', name: 'hours' },
            { data: 'is_billable', name: 'is_billable' },
            { data: 'status_badge', name: 'status' },
            { data: 'billing_amount', name: 'billing_amount', orderable: false },
            { data: 'actions', name: 'actions', orderable: false, searchable: false }
        ],
        order: [[0, 'desc']],
        responsive: true
    });

    // Approve timesheet
    window.approveTimesheet = function(id) {
        Swal.fire({
            title: pageData.labels.confirmApprove,
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: 'Yes',
            cancelButtonText: 'No'
        }).then((result) => {
            if (result.isConfirmed) {
                const url = pageData.urls.approve.replace(':id', id);
                
                $.ajax({
                    url: url,
                    method: 'POST',
                    success: function(response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: pageData.labels.approved,
                                showConfirmButton: true,
                                timer: 3000,
                                timerProgressBar: true
                            });
                            timesheetsTable.ajax.reload(null, false);
                        }
                    },
                    error: function(xhr) {
                        let errorMessage = 'An error occurred';
                        if (xhr.responseJSON) {
                            if (xhr.responseJSON.data) {
                                errorMessage = xhr.responseJSON.data;
                            } else if (xhr.responseJSON.message) {
                                errorMessage = xhr.responseJSON.message;
                            }
                        }
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: errorMessage,
                            showConfirmButton: true
                        });
                    }
                });
            }
        });
    };

    // Reject timesheet
    window.rejectTimesheet = function(id) {
        Swal.fire({
            title: pageData.labels.confirmReject,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Yes',
            cancelButtonText: 'No'
        }).then((result) => {
            if (result.isConfirmed) {
                const url = pageData.urls.reject.replace(':id', id);
                
                $.ajax({
                    url: url,
                    method: 'POST',
                    success: function(response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: pageData.labels.rejected,
                                showConfirmButton: true,
                                timer: 3000,
                                timerProgressBar: true
                            });
                            timesheetsTable.ajax.reload(null, false);
                        }
                    },
                    error: function(xhr) {
                        let errorMessage = 'An error occurred';
                        if (xhr.responseJSON) {
                            if (xhr.responseJSON.data) {
                                errorMessage = xhr.responseJSON.data;
                            } else if (xhr.responseJSON.message) {
                                errorMessage = xhr.responseJSON.message;
                            }
                        }
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: errorMessage,
                            showConfirmButton: true
                        });
                    }
                });
            }
        });
    };

    // Edit timesheet
    window.editTimesheet = function(id) {
        window.location.href = pageData.urls.create.replace('create', id + '/edit');
    };
    
    // Delete timesheet
    window.deleteTimesheet = function(id) {
        Swal.fire({
            title: 'Are you sure?',
            text: "You won't be able to revert this!",
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Yes, delete it!',
            cancelButtonText: 'No, cancel!'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.create.replace('create', id),
                    method: 'DELETE',
                    success: function(response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: 'Deleted!',
                                text: 'Timesheet has been deleted.',
                                showConfirmButton: true,
                                timer: 3000,
                                timerProgressBar: true
                            });
                            timesheetsTable.ajax.reload(null, false);
                        }
                    },
                    error: function(xhr) {
                        let errorMessage = 'An error occurred';
                        if (xhr.responseJSON) {
                            if (xhr.responseJSON.data) {
                                errorMessage = xhr.responseJSON.data;
                            } else if (xhr.responseJSON.message) {
                                errorMessage = xhr.responseJSON.message;
                            }
                        }
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: errorMessage,
                            showConfirmButton: true
                        });
                    }
                });
            }
        });
    };

    // Submit timesheet
    window.submitTimesheet = function(id) {
        Swal.fire({
            title: pageData.labels.confirmSubmit,
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: 'Yes',
            cancelButtonText: 'No'
        }).then((result) => {
            if (result.isConfirmed) {
                const url = pageData.urls.submit.replace(':id', id);
                
                $.ajax({
                    url: url,
                    method: 'POST',
                    success: function(response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: pageData.labels.submitted,
                                showConfirmButton: true,
                                timer: 3000,
                                timerProgressBar: true
                            });
                            timesheetsTable.ajax.reload(null, false);
                        }
                    },
                    error: function(xhr) {
                        let errorMessage = 'An error occurred';
                        if (xhr.responseJSON) {
                            if (xhr.responseJSON.data) {
                                errorMessage = xhr.responseJSON.data;
                            } else if (xhr.responseJSON.message) {
                                errorMessage = xhr.responseJSON.message;
                            }
                        }
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: errorMessage,
                            showConfirmButton: true
                        });
                    }
                });
            }
        });
    };
});