/**
 * Project Tasks Board (Kanban) JavaScript
 * PMCore Module - Task Board View
 */

$(function () {
    'use strict';

    // Setup CSRF token for AJAX requests
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Helper function to initialize all components
    function initializeComponents() {
        if (!window.pageData) {
            console.error('Board: initializeComponents called but pageData is not available');
            return;
        }

        console.log('Board: Initializing components with pageData...');
        
        // DOM elements
        const $kanbanBoard = $('#kanban-board');
        const $addTaskBtn = $('[data-bs-target="#offcanvasTaskForm"]'); // Updated to match the new ID
        
        // Initialize TaskFormHandler
        TaskFormHandler.init({
            formId: '#taskForm',
            offcanvasId: '#offcanvasTaskForm',
            offcanvasLabelId: '#offcanvasTaskFormLabel'
        });

        // Function definitions that use pageData and DOM elements
        function bindEvents() {
            // Add task button - use unified handler
            $addTaskBtn.on('click', function () {
                TaskFormHandler.showCreate();
            });

            // Edit task - use unified handler
            $(document).on('click', '.edit-task', function (e) {
                e.preventDefault();
                const taskId = $(this).data('id');
                TaskFormHandler.showEdit(taskId);
            });

            // Listen for task form success to refresh board
            $(document).on('taskFormSuccess', function(e, data) {
                loadKanbanBoard();
            });

            // Complete task
            $(document).on('click', '.complete-task', function (e) {
                e.preventDefault();
                const taskId = $(this).data('id');
                completeTask(taskId);
            });

            // Delete task
            $(document).on('click', '.delete-task', function (e) {
                e.preventDefault();
                const taskId = $(this).data('id');
                deleteTask(taskId);
            });
        }

        function loadKanbanBoard() {
            $kanbanBoard.html('<div class="text-center p-4"><i class="bx bx-loader-alt bx-spin"></i> ' + pageData.labels.loading + '</div>');

            $.ajax({
                url: pageData.urls.getDataAjax,
                method: 'GET',
                data: {
                    board_view: true
                },
                success: function (response) {
                    if (response.status === 'success') {
                        buildKanbanColumns(response.data);
                        initializeSortable();
                    } else {
                        showError(response.data || pageData.labels.error);
                    }
                },
                error: function () {
                    showError(pageData.labels.error);
                }
            });
        }

        function buildKanbanColumns(data) {
            let html = '';

            // Build columns for each status
            pageData.statuses.forEach(function (status) {
                const tasks = data.filter(task => task.task_status_id === status.id);
                
                html += `
                    <div class="kanban-column" data-status-id="${status.id}">
                        <div class="kanban-header d-flex justify-content-between align-items-center mb-3">
                            <h6 class="mb-0">${status.name}</h6>
                            <span class="badge bg-secondary">${tasks.length}</span>
                        </div>
                        <div class="kanban-tasks" data-status-id="${status.id}">
                            ${buildTaskCards(tasks)}
                        </div>
                    </div>
                `;
            });

            $kanbanBoard.html(html);
        }

        function buildTaskCards(tasks) {
            if (!tasks || tasks.length === 0) {
                return '<div class="text-muted text-center p-3">No tasks</div>';
            }

            return tasks.map(function (task) {
                const priorityClass = task.priority ? task.priority.color_class : 'secondary';
                const assignedUser = task.assigned_to ? task.assigned_to.name : 'Unassigned';
                const dueDateDisplay = task.due_date ? new Date(task.due_date).toLocaleDateString() : '';

                return `
                    <div class="kanban-card mb-3" data-id="${task.id}">
                        <div class="card task-card">
                            <div class="card-body p-3">
                                <div class="d-flex justify-content-between align-items-start mb-2">
                                    <h6 class="card-title mb-1">${escapeHtml(task.title)}</h6>
                                    <div class="dropdown">
                                        <button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">
                                            <i class="bx bx-dots-vertical-rounded"></i>
                                        </button>
                                        <ul class="dropdown-menu">
                                            <li><a class="dropdown-item edit-task" href="#" data-id="${task.id}">
                                                <i class="bx bx-edit me-1"></i> ${pageData.labels.editTask}
                                            </a></li>
                                            <li><a class="dropdown-item complete-task" href="#" data-id="${task.id}">
                                                <i class="bx bx-check me-1"></i> Complete
                                            </a></li>
                                            <li><hr class="dropdown-divider"></li>
                                            <li><a class="dropdown-item text-danger delete-task" href="#" data-id="${task.id}">
                                                <i class="bx bx-trash me-1"></i> Delete
                                            </a></li>
                                        </ul>
                                    </div>
                                </div>
                                
                                ${task.description ? `<p class="card-text small text-muted mb-2">${escapeHtml(task.description.substring(0, 100))}${task.description.length > 100 ? '...' : ''}</p>` : ''}
                                
                                <div class="d-flex justify-content-between align-items-center">
                                    <span class="badge bg-${priorityClass}">${task.priority ? task.priority.name : 'No Priority'}</span>
                                    <small class="text-muted">${assignedUser}</small>
                                </div>
                                
                                ${dueDateDisplay ? `<div class="mt-2"><small class="text-muted"><i class="bx bx-time me-1"></i> Due: ${dueDateDisplay}</small></div>` : ''}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function initializeSortable() {
            $('.kanban-tasks').each(function () {
                const element = this;
                new Sortable(element, {
                    group: 'kanban',
                    animation: 150,
                    ghostClass: 'sortable-ghost',
                    chosenClass: 'sortable-chosen',
                    dragClass: 'sortable-drag',
                    onEnd: function (evt) {
                        const taskId = $(evt.item).data('id');
                        const newStatusId = $(evt.to).data('status-id');
                        const newOrder = evt.newIndex; // Get the new position index
                        
                        if (taskId && newStatusId) {
                            updateTaskStatus(taskId, newStatusId, newOrder);
                        }
                    }
                });
            });
        }

        function updateTaskStatus(taskId, statusId, order = 0) {
            $.ajax({
                url: pageData.urls.reorder,
                method: 'POST',
                data: {
                    task_id: taskId,
                    task_status_id: statusId,
                    order: order
                },
                success: function (response) {
                    if (response.status === 'success') {
                        loadKanbanBoard(); // Refresh to update counts
                    } else {
                        showError(response.data || pageData.labels.error);
                        loadKanbanBoard(); // Revert on error
                    }
                },
                error: function (xhr) {
                    const errorMessage = xhr.responseJSON?.message || xhr.responseJSON?.data || pageData.labels.error;
                    showError(errorMessage);
                    loadKanbanBoard(); // Revert on error
                }
            });
        }

        function completeTask(taskId) {
            const completedStatus = pageData.statuses.find(s => s.name.toLowerCase() === 'completed');
            if (!completedStatus) {
                showError('Completed status not found');
                return;
            }

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
                    updateTaskStatus(taskId, completedStatus.id);
                }
            });
        }

        function deleteTask(taskId) {
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
                        url: pageData.urls.destroy?.replace('__ID__', taskId),
                        method: 'DELETE',
                        success: function (response) {
                            if (response.status === 'success') {
                                Swal.fire({
                                    icon: 'success',
                                    title: pageData.labels.deleteSuccess,
                                    timer: 2000,
                                    showConfirmButton: false
                                });
                                loadKanbanBoard();
                            } else {
                                showError(response.data || pageData.labels.error);
                            }
                        },
                        error: function () {
                            showError(pageData.labels.error);
                        }
                    });
                }
            });
        }

        function showError(message) {
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: message,
                customClass: {
                    confirmButton: 'btn btn-danger'
                },
                buttonsStyling: false
            });
        }

        function escapeHtml(text) {
            if (!text) return '';
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return text.replace(/[&<>"']/g, function (m) {
                return map[m];
            });
        }

        // Initialize components now that everything is defined
        loadKanbanBoard();
        bindEvents();
    }

    // Polling fallback function
    let retryCount = 0;
    function pollForPageData() {
        console.log('Board: Polling for pageData, attempt:', retryCount + 1, 'pageData available:', !!window.pageData);
        
        if (window.pageData) {
            console.log('Board: PageData found via polling, initializing components...');
            initializeComponents();
        } else {
            retryCount++;
            if (retryCount < 50) { // 5 seconds total
                setTimeout(pollForPageData, 100);
            } else {
                console.error('Board: PageData not available after 5 seconds. Check if pageData is defined in template.');
                console.log('Available global variables:', Object.keys(window).filter(key => key.includes('page') || key.includes('data')));
            }
        }
    }

    // Start initialization
    // Listen for pageDataReady event as primary method
    window.addEventListener('pageDataReady', function(event) {
        console.log('Board: PageData ready event received:', event.detail);
        initializeComponents();
    });
    
    // Fallback: try immediate initialization, then polling if needed
    if (window.pageData) {
        console.log('Board: PageData available immediately, initializing...');
        initializeComponents();
    } else {
        console.log('Board: PageData not available immediately, starting polling...');
        setTimeout(pollForPageData, 50); // Start polling after a short delay
    }
});
