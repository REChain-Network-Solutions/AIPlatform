$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Variables that will be initialized when pageData is available
    let tasksTable;

    // Helper function to do the actual initialization
    function initializeComponents() {
        if (window.pageData) {
            console.log('Initializing components with pageData...');

            // Initialize TaskFormHandler
            TaskFormHandler.init({
                formId: '#taskForm',
                offcanvasId: '#offcanvasTaskForm',
                offcanvasLabelId: '#offcanvasTaskFormLabel'
            });

            // Initialize DataTables
            initializeDataTables();

            // Bind event handlers
            bindEventHandlers();
        } else {
            console.error('initializeComponents called but pageData is not available');
        }
    }

    // Polling fallback function
    let retryCount = 0;
    function pollForPageData() {
        console.log('Polling for pageData, attempt:', retryCount + 1, 'pageData available:', !!window.pageData);

        if (window.pageData) {
            console.log('PageData found via polling, initializing components...');
            initializeComponents();
        } else {
            retryCount++;
            if (retryCount < 50) { // 5 seconds total
                setTimeout(pollForPageData, 100);
            } else {
                console.error('PageData not available after 5 seconds. Check if pageData is defined in template.');
                console.log('Available global variables:', Object.keys(window).filter(key => key.includes('page') || key.includes('data')));
            }
        }
    }

    // Initialize DataTables
    function initializeDataTables() {
        tasksTable = $('.datatables-tasks').DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: pageData.urls.tasksData,
                type: 'GET'
            },
            columns: [
                { data: 'title', name: 'title' },
                { data: 'status', name: 'status.name' },
                { data: 'priority', name: 'priority.name' },
                { data: 'assigned_to', name: 'assigned_to', orderable: false, searchable: false },
                { data: 'due_date', name: 'due_date' },
                { data: 'actions', name: 'actions', orderable: false, searchable: false }
            ],
            order: [[0, 'desc']],
            dom: '<"row"<"col-md-2"<"me-3"l>><"col-md-10"<"dt-action-buttons text-xl-end text-lg-start text-md-end text-start d-flex align-items-center justify-content-end flex-md-row flex-column mb-3 mb-md-0"fB>>>t<"row"<"col-sm-12 col-md-6"i><"col-sm-12 col-md-6"p>>',
            displayLength: 10,
            lengthMenu: [10, 25, 50, 75, 100],
            buttons: [
                {
                    extend: 'collection',
                    className: 'btn btn-outline-secondary dropdown-toggle mx-3',
                    text: '<i class="bx bx-export me-1"></i>Export',
                    buttons: [
                        {
                            extend: 'print',
                            text: '<i class="bx bx-printer me-1" ></i>Print',
                            className: 'dropdown-item'
                        },
                        {
                            extend: 'csv',
                            text: '<i class="bx bx-file me-1" ></i>Csv',
                            className: 'dropdown-item'
                        },
                        {
                            extend: 'excel',
                            text: '<i class="bx bx-file me-1" ></i>Excel',
                            className: 'dropdown-item'
                        },
                        {
                            extend: 'pdf',
                            text: '<i class="bx bx-file-pdf me-1" ></i>Pdf',
                            className: 'dropdown-item'
                        }
                    ]
                }
            ],
            responsive: {
                details: {
                    display: $.fn.dataTable.Responsive.display.modal({
                        header: function (row) {
                            var data = row.data();
                            return 'Details for ' + data['title'];
                        }
                    }),
                    type: 'column',
                    renderer: function (api, rowIdx, columns) {
                        var data = $.map(columns, function (col, i) {
                            return col.title !== ''
                                ? '<tr data-dt-row="' +
                                    col.rowIndex +
                                    '" data-dt-column="' +
                                    col.columnIndex +
                                    '">' +
                                    '<td>' +
                                    col.title +
                                    ':' +
                                    '</td> ' +
                                    '<td>' +
                                    col.data +
                                    '</td>' +
                                    '</tr>'
                                : '';
                        }).join('');

                        return data ? $('<table class="table"/><tbody />').append(data) : false;
                    }
                }
            },
            language: {
                searchPlaceholder: 'Search tasks...',
                search: '',
                lengthMenu: '_MENU_',
                processing: '<div class="d-flex justify-content-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>'
            }
        });
    }

    // Bind all event handlers
    function bindEventHandlers() {
        // Listen for task form success events to reload table
        $(document).on('taskFormSuccess', function(e, data) {
            tasksTable.ajax.reload();
        });

        // Add task button click handler
        $(document).on('click', '[data-bs-target="#offcanvasTaskForm"]', function () {
            TaskFormHandler.showCreate();
        });

        // Edit task
        $(document).on('click', '.edit-task', function () {
            const taskId = $(this).data('id');
            TaskFormHandler.showEdit(taskId);
        });

        // Complete task
        $(document).on('click', '.complete-task', function () {
            const taskId = $(this).data('id');

            Swal.fire({
                title: pageData.labels.completeConfirm,
                icon: 'question',
                showCancelButton: true,
                confirmButtonText: 'Yes, complete it!',
                cancelButtonText: 'Cancel',
                customClass: {
                    confirmButton: 'btn btn-success',
                    cancelButton: 'btn btn-outline-secondary'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    $.ajax({
                        url: pageData.urls.tasksComplete.replace('__ID__', taskId),
                        method: 'POST',
                        success: function (response) {
                            if (response.status === 'success') {
                                Swal.fire({
                                    icon: 'success',
                                    title: pageData.labels.completeSuccess,
                                    timer: 2000,
                                    showConfirmButton: false
                                });

                                tasksTable.ajax.reload();
                            }
                        },
                        error: function (xhr) {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: xhr.responseJSON?.message || pageData.labels.error
                            });
                        }
                    });
                }
            });
        });

        // Delete task
        $(document).on('click', '.delete-task', function () {
            const taskId = $(this).data('id');

            Swal.fire({
                title: pageData.labels.confirmDelete,
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: 'Yes, delete it!',
                cancelButtonText: 'Cancel',
                customClass: {
                    confirmButton: 'btn btn-danger',
                    cancelButton: 'btn btn-outline-secondary'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    $.ajax({
                        url: pageData.urls.tasksDestroy.replace('__ID__', taskId),
                        method: 'DELETE',
                        success: function (response) {
                            if (response.status === 'success') {
                                Swal.fire({
                                    icon: 'success',
                                    title: pageData.labels.deleteSuccess,
                                    timer: 2000,
                                    showConfirmButton: false
                                });

                                tasksTable.ajax.reload();
                            }
                        },
                        error: function (xhr) {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: xhr.responseJSON?.message || pageData.labels.error
                            });
                        }
                    });
                }
            });
        });

        // Start task
        $(document).on('click', '.start-time', function () {
            const taskId = $(this).data('id');

            Swal.fire({
                title: pageData.labels.startConfirm,
                icon: 'question',
                showCancelButton: true,
                confirmButtonText: 'Yes, start it!',
                cancelButtonText: 'Cancel',
                customClass: {
                    confirmButton: 'btn btn-success',
                    cancelButton: 'btn btn-outline-secondary'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    $.ajax({
                        url: pageData.urls.tasksStart.replace('__ID__', taskId),
                        method: 'POST',
                        success: function (response) {
                            if (response.status === 'success') {
                                Swal.fire({
                                    icon: 'success',
                                    title: pageData.labels.startSuccess,
                                    timer: 2000,
                                    showConfirmButton: false
                                });

                                tasksTable.ajax.reload();
                            }
                        },
                        error: function (xhr) {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: xhr.responseJSON?.message || pageData.labels.error
                            });
                        }
                    });
                }
            });
        });

        // Stop task
        $(document).on('click', '.stop-time', function () {
            const taskId = $(this).data('id');

            Swal.fire({
                title: pageData.labels.stopConfirm,
                icon: 'question',
                showCancelButton: true,
                confirmButtonText: 'Yes, stop it!',
                cancelButtonText: 'Cancel',
                customClass: {
                    confirmButton: 'btn btn-warning',
                    cancelButton: 'btn btn-outline-secondary'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    $.ajax({
                        url: pageData.urls.tasksStop.replace('__ID__', taskId),
                        method: 'POST',
                        success: function (response) {
                            if (response.status === 'success') {
                                Swal.fire({
                                    icon: 'success',
                                    title: pageData.labels.stopSuccess,
                                    timer: 2000,
                                    showConfirmButton: false
                                });

                                tasksTable.ajax.reload();
                            }
                        },
                        error: function (xhr) {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: xhr.responseJSON?.message || pageData.labels.error
                            });
                        }
                    });
                }
            });
        });
    }

    // Start initialization
    // Listen for pageDataReady event as primary method
    window.addEventListener('pageDataReady', function(event) {
        console.log('PageData ready event received:', event.detail);
        initializeComponents();
    });

    // Fallback: try immediate initialization, then polling if needed
    if (window.pageData) {
        console.log('PageData available immediately, initializing...');
        initializeComponents();
    } else {
        console.log('PageData not available immediately, starting polling...');
        setTimeout(pollForPageData, 50); // Start polling after a short delay
    }

    // Helper function to do the actual initialization
    function initializeComponents() {
        if (window.pageData) {
            console.log('Initializing components with pageData...');

            // Initialize TaskFormHandler
            TaskFormHandler.init({
                formId: '#taskForm',
                offcanvasId: '#offcanvasTaskForm',
                offcanvasLabelId: '#offcanvasTaskFormLabel'
            });

            // Initialize DataTables
            initializeDataTables();

            // Bind event handlers
            bindEventHandlers();
        }
    }
});
