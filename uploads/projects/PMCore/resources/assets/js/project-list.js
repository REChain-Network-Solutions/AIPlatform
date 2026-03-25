$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Initialize DataTables
    const projectsTable = $('.datatables-projects').DataTable({
        processing: true,
        serverSide: true,
        ajax: {
            url: window.pageData.urls.projectData,
            type: 'GET'
        },
        columns: [
            { data: 'id', name: 'id' },
            { data: 'name', name: 'name' },
            { data: 'code', name: 'code' },
            { data: 'client_name', name: 'client.name', orderable: false },
            { data: 'status_badge', name: 'status', orderable: false },
            { data: 'priority_badge', name: 'priority', orderable: false },
            { data: 'start_date', name: 'start_date' },
            { data: 'end_date', name: 'end_date' },
            { data: 'progress', name: 'progress', orderable: false, searchable: false },
            { data: 'actions', name: 'actions', orderable: false, searchable: false }
        ],
        order: [[0, 'desc']],
        dom: '<"card-header d-flex flex-wrap pb-0 pt-3"<"me-5 ms-n2"f><"d-flex justify-content-start justify-content-md-end align-items-baseline"<"dt-action-buttons d-flex flex-column align-items-start align-items-md-center justify-content-md-end gap-3 gap-md-0 pt-0"lB>>>t<"row"<"col-sm-12 col-md-6"i><"col-sm-12 col-md-6"p>>',
        displayLength: 10,
        lengthMenu: [10, 25, 50, 75, 100],
        language: {
            search: window.pageData.labels.search,
            searchPlaceholder: window.pageData.labels.searchPlaceholder,
            lengthMenu: window.pageData.labels.lengthMenu,
            info: window.pageData.labels.info,
            infoEmpty: window.pageData.labels.infoEmpty,
            infoFiltered: window.pageData.labels.infoFiltered,
            loadingRecords: window.pageData.labels.loadingRecords,
            processing: window.pageData.labels.processing,
            zeroRecords: window.pageData.labels.zeroRecords,
            emptyTable: window.pageData.labels.emptyTable,
            paginate: {
                first: window.pageData.labels.paginate.first,
                last: window.pageData.labels.paginate.last,
                next: '<i class="bx bx-chevron-right bx-18px"></i>',
                previous: '<i class="bx bx-chevron-left bx-18px"></i>'
            }
        },
        buttons: [],
        responsive: true,
        drawCallback: function () {
            // Re-initialize tooltips if used
            $('[data-bs-toggle="tooltip"]').tooltip();
        }
    });

    // Make projectsTable available globally
    window.projectsTable = projectsTable;
});

// Delete project (global function)
window.deleteProject = function(projectId) {
    Swal.fire({
        title: window.pageData.labels.confirmDelete,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: window.pageData.labels.yesDeleteIt,
        cancelButtonText: window.pageData.labels.cancel,
        customClass: {
            confirmButton: 'btn btn-danger me-3',
            cancelButton: 'btn btn-label-secondary'
        },
        buttonsStyling: false
    }).then((result) => {
        if (result.isConfirmed) {
            $.ajax({
                url: window.pageData.urls.projectDelete.replace('__ID__', projectId),
                method: 'DELETE',
                success: function (response) {
                    if (response.status === 'success') {
                        Swal.fire({
                            icon: 'success',
                            title: window.pageData.labels.deleted,
                            text: window.pageData.labels.deleteSuccess,
                            timer: 2000,
                            showConfirmButton: false
                        });

                        window.projectsTable.ajax.reload();
                    } else {
                        Swal.fire({
                            icon: 'error',
                            title: 'Error',
                            text: response.data || response.message || window.pageData.labels.error
                        });
                    }
                },
                error: function (xhr) {
                    Swal.fire({
                        icon: 'error',
                        title: 'Error',
                        text: xhr.responseJSON?.message || window.pageData.labels.error
                    });
                }
            });
        }
    });
};
