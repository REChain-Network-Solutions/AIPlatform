$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Initialize variables
    let timesheetTable;

    // Initialize components
    initializeDataTable();
    initializeFilters();
    initializeFiltersForm();
    loadStatistics();

    /**
     * Initialize DataTable with timesheets
     */
    function initializeDataTable() {
        timesheetTable = $('.datatables-timesheets').DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: pageData.urls.datatableUrl,
                type: 'GET',
                data: function (d) {
                    d.user_id = $('#filterUser').val();
                    d.project_id = $('#filterProject').val();
                    d.status = $('#filterStatus').val();
                    d.date_from = $('#filterDateFrom').val();
                    d.date_to = $('#filterDateTo').val();
                }
            },
            columns: [
                { data: 'user', name: 'user.name', orderable: false },
                { data: 'project_name', name: 'project.name' },
                { data: 'task_name', name: 'task.title', orderable: false, searchable: false },
                { data: 'formatted_date', name: 'date' },
                { data: 'formatted_hours', name: 'hours', className: 'text-end' },
                { data: 'description', name: 'description' },
                { data: 'billing_amount', name: 'billing_amount', className: 'text-end', orderable: false, searchable: false },
                { data: 'status_badge', name: 'status', className: 'text-center' },
                { data: 'approved_by', name: 'approved_by.name', orderable: false, searchable: false },
                { data: 'actions', name: 'actions', orderable: false, searchable: false, className: 'text-center' }
            ],
            order: [[3, 'desc']],
            displayLength: 25,
            lengthMenu: [10, 25, 50, 75, 100],
            language: {
                sLengthMenu: '_MENU_',
                search: '',
                searchPlaceholder: 'Search Timesheets'
            }
        });
    }

    /**
     * Initialize filters
     */
    function initializeFilters() {
        // Initialize user filter
        $('#filterUser').select2({
            placeholder: pageData.labels.allUsers || 'All Users',
            allowClear: true,
            ajax: {
                url: pageData.urls.usersSearchUrl,
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        search: params.term,
                        page: params.page
                    };
                },
                processResults: function (data) {
                    return {
                        results: data.data.map(function (user) {
                            return {
                                id: user.id,
                                text: user.name
                            };
                        })
                    };
                },
                cache: true
            }
        });

        // Initialize project filter
        $('#filterProject').select2({
            placeholder: pageData.labels.allProjects || 'All Projects',
            allowClear: true,
            ajax: {
                url: pageData.urls.projectsSearchUrl,
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        search: params.term,
                        page: params.page
                    };
                },
                processResults: function (data) {
                    return {
                        results: data.data.map(function (project) {
                            return {
                                id: project.id,
                                text: project.name
                            };
                        })
                    };
                },
                cache: true
            }
        });

        // Initialize date pickers
        $('#filterDateFrom, #filterDateTo').flatpickr({
            dateFormat: 'Y-m-d',
            allowInput: true
        });
    }

    /**
     * Initialize filters form
     */
    function initializeFiltersForm() {
        $('#filterForm').on('submit', function (e) {
            e.preventDefault();
            timesheetTable.ajax.reload();
            loadStatistics();
        });

        $('#clearFilters').on('click', function () {
            $('#filterForm')[0].reset();
            $('#filterUser').val(null).trigger('change');
            $('#filterProject').val(null).trigger('change');
            $('#filterStatus').val('');
            $('#filterDateFrom').val('');
            $('#filterDateTo').val('');
            timesheetTable.ajax.reload();
            loadStatistics();
        });
    }

    /**
     * Load statistics
     */
    function loadStatistics() {
        const filters = {
            user_id: $('#filterUser').val(),
            project_id: $('#filterProject').val(),
            status: $('#filterStatus').val(),
            date_from: $('#filterDateFrom').val(),
            date_to: $('#filterDateTo').val()
        };

        $.ajax({
            url: pageData.urls.statisticsUrl,
            method: 'GET',
            data: filters,
            success: function (response) {
                if (response.status === 'success') {
                    const stats = response.data.statistics;
                    $('#totalHours').text(parseFloat(stats.total_hours || 0).toFixed(2));
                    $('#billableHours').text(parseFloat(stats.billable_hours || 0).toFixed(2));
                    $('#approvedHours').text(parseFloat(stats.approved_hours || 0).toFixed(2));
                    $('#pendingHours').text(parseFloat(stats.pending_hours || 0).toFixed(2));
                }
            },
            error: function () {
                console.error('Failed to load statistics');
            }
        });
    }

    /**
     * Show notification
     */
    function showNotification(type, message) {
        Swal.fire({
            icon: type,
            title: type === 'success' ? 'Success!' : 'Error!',
            text: message,
            showConfirmButton: true,
            confirmButtonText: 'OK',
            timer: type === 'success' ? 3000 : null,
            timerProgressBar: type === 'success' ? true : false
        });
    }

    // Global functions for button actions
    window.editTimesheet = function (id) {
        window.location.href = pageData.urls.editUrl.replace('__ID__', id);
    };

    window.deleteTimesheet = function (id) {
        Swal.fire({
            title: 'Are you sure?',
            text: pageData.labels.confirmDelete,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#3085d6',
            confirmButtonText: 'Yes, delete it!'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.deleteUrl.replace('__ID__', id),
                    method: 'DELETE',
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.deleteSuccess);
                            timesheetTable.ajax.reload();
                            loadStatistics();
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

    window.approveTimesheet = function (id) {
        Swal.fire({
            title: 'Are you sure?',
            text: pageData.labels.confirmApprove,
            icon: 'question',
            showCancelButton: true,
            confirmButtonColor: '#28a745',
            cancelButtonColor: '#6c757d',
            confirmButtonText: 'Yes, approve it!'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.approveUrl.replace('__ID__', id),
                    method: 'POST',
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.approveSuccess);
                            timesheetTable.ajax.reload();
                            loadStatistics();
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

    window.rejectTimesheet = function (id) {
        Swal.fire({
            title: 'Are you sure?',
            text: pageData.labels.confirmReject,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#dc3545',
            cancelButtonColor: '#6c757d',
            confirmButtonText: 'Yes, reject it!'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.rejectUrl.replace('__ID__', id),
                    method: 'POST',
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.rejectSuccess);
                            timesheetTable.ajax.reload();
                            loadStatistics();
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

    window.submitTimesheet = function (id) {
        Swal.fire({
            title: 'Are you sure?',
            text: pageData.labels.confirmSubmit,
            icon: 'question',
            showCancelButton: true,
            confirmButtonColor: '#007bff',
            cancelButtonColor: '#6c757d',
            confirmButtonText: 'Yes, submit it!'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.submitUrl.replace('__ID__', id),
                    method: 'POST',
                    success: function (response) {
                        if (response.status === 'success') {
                            showNotification('success', pageData.labels.submitSuccess);
                            timesheetTable.ajax.reload();
                            loadStatistics();
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
});
