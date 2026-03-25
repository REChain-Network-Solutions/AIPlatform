$(function () {
    'use strict';

    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    let projectStatusTable;
    let isEditMode = false;
    let currentStatusId = null;
    const formOffcanvas = new bootstrap.Offcanvas(document.getElementById('projectStatusFormOffcanvas'));

    initializeDataTable();
    projectStatusTable.on('draw', function() {
        initializeSortable();
    });
    
    $('#projectStatusForm').on('submit', function (e) {
        e.preventDefault();
        handleFormSubmission();
    });
    
    $('#projectStatusFormOffcanvas').on('hidden.bs.offcanvas', function () {
        resetForm();
    });
    
    function initializeDataTable() {
        projectStatusTable = $('.datatables-project-statuses').DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: pageData.urls.datatableUrl,
                type: 'GET'
            },
            columns: [
                { data: 'sort_order', name: 'sort_order', width: '5%', className: 'text-center' },
                { data: 'name', name: 'name', orderable: false },
                { data: 'description', name: 'description', orderable: false },
                { data: 'projects_count', name: 'projects_count', className: 'text-center', orderable: false },
                { data: 'is_active', name: 'is_active', className: 'text-center', orderable: false },
                { data: 'is_default', name: 'is_default', className: 'text-center', orderable: false },
                { data: 'is_completed', name: 'is_completed', className: 'text-center', orderable: false },
                { data: 'actions', name: 'actions', orderable: false, searchable: false, className: 'text-center' }
            ],
            order: [[0, 'asc']],
            dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6 d-flex justify-content-center justify-content-md-end"f>>t<"row"<"col-sm-12 col-md-6"i><"col-sm-12 col-md-6"p>>',
            displayLength: 25,
            lengthMenu: [10, 25, 50, 75, 100],
            language: {
                sLengthMenu: '_MENU_',
                search: '',
                searchPlaceholder: pageData.labels.searchPlaceholder || 'Search...'
            }
        });
    }
    
    function initializeSortable() {
        const tableBody = document.querySelector('.datatables-project-statuses tbody');
        
        if (tableBody) {
            if (tableBody.sortable) {
                tableBody.sortable.destroy();
            }
            tableBody.sortable = new Sortable(tableBody, {
                animation: 150,
                handle: '.drag-handle',
                ghostClass: 'sortable-ghost',
                onEnd: function (evt) {
                    updateSortOrder();
                }
            });
        }
    }

    function updateSortOrder() {
        const statusIds = [];
        $('.datatables-project-statuses tbody tr').each(function () {
            const rowData = projectStatusTable.row(this).data();
            if (rowData && rowData.id) {
                statusIds.push(rowData.id);
            }
        });

        if (statusIds.length > 0) {
            $.ajax({
                url: pageData.urls.sortOrderUrl,
                method: 'POST',
                data: {
                    status_ids: statusIds
                },
                success: function (response) {
                    if (response.status === 'success') {
                        showNotification('success', pageData.labels.sortSuccess);
                        setTimeout(function() {
                            projectStatusTable.ajax.reload(null, false);
                        }, 500);
                    } else {
                        showNotification('error', pageData.labels.error);
                    }
                },
                error: function () {
                    showNotification('error', pageData.labels.error);
                    projectStatusTable.ajax.reload(null, false);
                }
            });
        }
    }

    function handleFormSubmission() {
        const form = $('#projectStatusForm')[0];
        const formData = new FormData(form);

        const checkboxes = ['is_active', 'is_default', 'is_completed'];
        checkboxes.forEach(function (checkbox) {
            const isChecked = $('#' + checkbox.replace('is_', 'status')).is(':checked');
            formData.delete(checkbox);
            formData.append(checkbox, isChecked ? '1' : '0');
        });

        const url = isEditMode ? 
            pageData.urls.updateUrl.replace('__ID__', currentStatusId) : 
            pageData.urls.storeUrl;
        
        if (isEditMode) {
            formData.append('_method', 'PUT');
        }

        $.ajax({
            url: url,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response.status === 'success') {
                    const message = isEditMode ? pageData.labels.updateSuccess : pageData.labels.createSuccess;
                    showNotification('success', message);
                    formOffcanvas.hide();
                    projectStatusTable.ajax.reload();
                } else {
                    handleFormErrors(response.errors || {});
                }
            },
            error: function (xhr) {
                if (xhr.status === 422) {
                    const errors = xhr.responseJSON.errors || {};
                    handleFormErrors(errors);
                } else {
                    showNotification('error', pageData.labels.error);
                }
            }
        });
    }

    function handleFormErrors(errors) {
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');

        Object.keys(errors).forEach(function (field) {
            const fieldName = field.replace('is_', 'status');
            const input = $(`#${fieldName}`);
            const feedback = input.siblings('.invalid-feedback');
            
            if (input.length) {
                input.addClass('is-invalid');
                feedback.text(errors[field][0]);
            }
        });

        showNotification('error', pageData.labels.validationError);
    }

    function resetForm() {
        $('#projectStatusForm')[0].reset();
        $('#statusId').val('');
        $('#statusColor').val('#007bff');
        $('#statusActive').prop('checked', true);
        $('#statusDefault').prop('checked', false);
        $('#statusCompleted').prop('checked', false);
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');
        isEditMode = false;
        currentStatusId = null;
        $('#offcanvasTitle').text(pageData.labels.addTitle || 'Add Project Status');
    }

    function showNotification(type, message) {
        Swal.fire({
            icon: type,
            title: type === 'success' ? (pageData.labels.successTitle || 'Success!') : (pageData.labels.errorTitle || 'Error!'),
            text: message,
            showConfirmButton: true,
            confirmButtonText: pageData.labels.okButtonText || 'OK',
            timer: type === 'success' ? 3000 : null,
            timerProgressBar: type === 'success' ? true : false
        });
    }
    
    // Global functions for actions
    window.editStatus = function(id) {
        isEditMode = true;
        currentStatusId = id;
        $('#offcanvasTitle').text(pageData.labels.editTitle || 'Edit Project Status');
        
        $.ajax({
            url: pageData.urls.updateUrl.replace('__ID__', id),
            method: 'GET',
            success: function (response) {
                if (response.status === 'success') {
                    const status = response.data.status;
                    $('#statusId').val(status.id);
                    $('#statusName').val(status.name);
                    $('#statusDescription').val(status.description);
                    $('#statusColor').val(status.color);
                    $('#statusActive').prop('checked', status.is_active);
                    $('#statusDefault').prop('checked', status.is_default);
                    $('#statusCompleted').prop('checked', status.is_completed);
                    
                    formOffcanvas.show();
                } else {
                    showNotification('error', pageData.labels.error);
                }
            },
            error: function () {
                showNotification('error', pageData.labels.error);
            }
        });
    };

    window.deleteStatus = function(id) {
        Swal.fire({
            title: pageData.labels.confirmTitle || 'Are you sure?',
            text: pageData.labels.confirmDelete,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#3085d6',
            confirmButtonText: pageData.labels.confirmButtonText || 'Yes, delete it!',
            cancelButtonText: pageData.labels.cancelButtonText || 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.deleteUrl.replace('__ID__', id),
                    method: 'DELETE',
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.deleteSuccess);
                            projectStatusTable.ajax.reload();
                        } else {
                            showNotification('error', response.data || pageData.labels.error);
                        }
                    },
                    error: function (xhr) {
                        const message = xhr.responseJSON?.data || pageData.labels.error;
                        showNotification('error', message);
                    }
                });
            }
        });
    };

    window.toggleStatus = function(id) {
        Swal.fire({
            title: pageData.labels.confirmTitle || 'Are you sure?',
            text: pageData.labels.confirmToggle,
            icon: 'question',
            showCancelButton: true,
            confirmButtonColor: '#3085d6',
            cancelButtonColor: '#d33',
            confirmButtonText: pageData.labels.toggleConfirmButtonText || 'Yes, change it!',
            cancelButtonText: pageData.labels.cancelButtonText || 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.toggleActiveUrl.replace('__ID__', id),
                    method: 'POST',
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.toggleSuccess);
                            projectStatusTable.ajax.reload();
                        } else {
                            showNotification('error', pageData.labels.error);
                        }
                    },
                    error: function () {
                        showNotification('error', pageData.labels.error);
                    }
                });
            }
        });
    };

    window.setDefaultStatus = function(id) {
        Swal.fire({
            title: pageData.labels.confirmTitle || 'Are you sure?',
            text: pageData.labels.confirmSetDefault || 'This will set this status as the default for new projects.',
            icon: 'question',
            showCancelButton: true,
            confirmButtonColor: '#3085d6',
            cancelButtonColor: '#d33',
            confirmButtonText: pageData.labels.setDefaultConfirmButtonText || 'Yes, set as default!',
            cancelButtonText: pageData.labels.cancelButtonText || 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.setDefaultUrl,
                    method: 'POST',
                    data: {
                        id: id
                    },
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.setDefaultSuccess || 'Status set as default successfully!');
                            projectStatusTable.ajax.reload();
                        } else {
                            showNotification('error', pageData.labels.error);
                        }
                    },
                    error: function () {
                        showNotification('error', pageData.labels.error);
                    }
                });
            }
        });
    };
});