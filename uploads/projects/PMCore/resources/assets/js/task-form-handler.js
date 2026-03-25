/**
 * Unified Task Form Handler
 * Shared between list and board views
 */

window.TaskFormHandler = (function() {
    'use strict';

    // Private variables
    let $form, $offcanvas, $submitBtn, $title, $description, $status, $priority, $assignedTo, $dueDate, $estimatedHours, $milestone;
    let flatpickrInstance;
    let isInitialized = false;

    // Configuration
    const config = {
        formId: '#taskForm',
        offcanvasId: '#offcanvasTaskForm',
        offcanvasLabelId: '#offcanvasTaskFormLabel',
        submitBtnId: '.data-submit'
    };

    // Initialize DOM elements
    function initializeElements() {
        $form = $(config.formId);
        $offcanvas = $(config.offcanvasId);
        $submitBtn = $form.find(config.submitBtnId);
        $title = $('#task-title');
        $description = $('#task-description');
        $status = $('#task-status');
        $priority = $('#task-priority');
        $assignedTo = $('#task-assigned-to');
        $dueDate = $('#task-due-date');
        $estimatedHours = $('#task-estimated-hours');
        $milestone = $('#task-is-milestone');

        // Alternative selectors for different naming conventions
        if ($title.length === 0) $title = $('#task_title');
        if ($description.length === 0) $description = $('#task_description');
        if ($status.length === 0) $status = $('#task_status_id');
        if ($priority.length === 0) $priority = $('#task_priority_id');
        if ($assignedTo.length === 0) $assignedTo = $('#assigned_to_user_id');
        if ($dueDate.length === 0) $dueDate = $('#task_due_date');
        if ($estimatedHours.length === 0) $estimatedHours = $('#task_estimated_hours');
        if ($milestone.length === 0) $milestone = $('#task_is_milestone');
    }

    // Initialize Flatpickr
    function initializeFlatpickr() {
        if ($dueDate.length && typeof flatpickr !== 'undefined') {
            // Destroy existing instance if it exists
            if (flatpickrInstance) {
                flatpickrInstance.destroy();
            }

            flatpickrInstance = $dueDate[0]._flatpickr || $dueDate.flatpickr({
                dateFormat: 'Y-m-d',
                allowInput: true
            });
        }
    }

    // Initialize Select2
    function initializeSelect2() {
        if (typeof $.fn.select2 !== 'undefined') {
            $assignedTo.each(function() {
                if (!$(this).hasClass('select2-hidden-accessible')) {
                    $(this).select2({
                        dropdownParent: $offcanvas,
                        width: '100%',
                        placeholder: 'Select User'
                    });
                }
            });
        }
    }

    // Format date for input
    function formatDateForInput(dateString) {
        if (!dateString) return '';

        // If already in Y-m-d format, return as is
        if (/^\d{4}-\d{2}-\d{2}$/.test(dateString)) {
            return dateString;
        }

        // Extract date part from datetime strings
        if (dateString.length >= 10) {
            const datePart = dateString.substring(0, 10);
            if (/^\d{4}-\d{2}-\d{2}$/.test(datePart)) {
                return datePart;
            }
        }

        return dateString;
    }

    // Get pageData safely
    function getPageData() {
        return window.pageData || {};
    }

    // Get labels safely
    function getLabels() {
        const pageData = getPageData();
        return pageData.labels || {};
    }

    // Get URLs safely
    function getUrls() {
        const pageData = getPageData();
        return pageData.urls || {};
    }

    // Reset form to initial state
    function resetForm() {
        $form[0].reset();

        // Clear Select2
        if ($assignedTo.hasClass('select2-hidden-accessible')) {
            $assignedTo.val('').trigger('change');
        }

        // Clear Flatpickr
        if (flatpickrInstance) {
            flatpickrInstance.clear();
        }

        // Clear validation errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').remove();

        // Reset form mode
        $form.removeData('task-id');
        const labels = getLabels();
        $(config.offcanvasLabelId).text(labels.addTask || 'Add Task');
    }

    // Populate form with task data for editing
    function populateForm(task) {
        console.log('TaskFormHandler: Populating form with task data:', task);
        
        $title.val(task.title || '');
        $description.val(task.description || '');
        $status.val(task.task_status_id || '').trigger('change');
        $priority.val(task.task_priority_id || '').trigger('change');
        $assignedTo.val(task.assigned_to_user_id || '').trigger('change');
        $estimatedHours.val(task.estimated_hours || '');
        $milestone.prop('checked', !!task.is_milestone);

        // Set due date
        const formattedDate = formatDateForInput(task.due_date);
        if (formattedDate && flatpickrInstance) {
            flatpickrInstance.setDate(formattedDate);
        } else {
            $dueDate.val(formattedDate);
        }

        // Set form to edit mode
        $form.data('task-id', task.id);
        const labels = getLabels();
        $(config.offcanvasLabelId).text(labels.editTask || 'Edit Task');
        
        console.log('TaskFormHandler: Form populated successfully');
    }

    // Handle form submission
    function handleSubmit(e) {
        e.preventDefault();

        const taskId = $form.data('task-id');
        const isEdit = !!taskId;
        const urls = getUrls();
        const labels = getLabels();

        const url = isEdit ?
            urls.tasksUpdate?.replace('__ID__', taskId) :
            urls.tasksStore;

        if (!url) {
            console.error('TaskFormHandler: Missing URL configuration');
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: 'Configuration error. Please refresh the page and try again.'
            });
            return;
        }

        const formData = new FormData($form[0]);

        // Add method override for PUT requests
        if (isEdit) {
            formData.append('_method', 'PUT');
        }

        // Fix checkbox value
        const isMilestoneChecked = $milestone.is(':checked');
        formData.delete('is_milestone');
        formData.append('is_milestone', isMilestoneChecked ? '1' : '0');

        // Clear validation errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').remove();

        $.ajax({
            url: url,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'success') {
                    $offcanvas.offcanvas('hide');

                    // Show success message
                    Swal.fire({
                        icon: 'success',
                        title: 'Success',
                        text: response.data.message || (isEdit ?
                            labels.taskUpdated || 'Task updated successfully!' :
                            labels.taskCreated || 'Task created successfully!'),
                        timer: 2000,
                        showConfirmButton: false
                    });

                    // Trigger custom event for other components to listen
                    $(document).trigger('taskFormSuccess', {
                        action: isEdit ? 'update' : 'create',
                        task: response.data.task
                    });
                } else {
                    handleValidationErrors(response.data);
                }
            },
            error: function(xhr) {
                if (xhr.responseJSON && xhr.responseJSON.errors) {
                    handleValidationErrors(xhr.responseJSON.errors);
                } else {
                    Swal.fire({
                        icon: 'error',
                        title: 'Error',
                        text: xhr.responseJSON?.message || labels.error || 'An error occurred'
                    });
                }
            }
        });
    }

    // Handle validation errors
    function handleValidationErrors(errors) {
        $.each(errors, function(key, messages) {
            let fieldElement;

            // Map field names to form elements
            switch(key) {
                case 'title':
                    fieldElement = $title;
                    break;
                case 'description':
                    fieldElement = $description;
                    break;
                case 'task_status_id':
                    fieldElement = $status;
                    break;
                case 'task_priority_id':
                    fieldElement = $priority;
                    break;
                case 'assigned_to_user_id':
                    fieldElement = $assignedTo;
                    break;
                case 'due_date':
                    fieldElement = $dueDate;
                    break;
                case 'estimated_hours':
                    fieldElement = $estimatedHours;
                    break;
                default:
                    fieldElement = $('[name="' + key + '"]');
            }

            if (fieldElement && fieldElement.length) {
                fieldElement.addClass('is-invalid');
                fieldElement.parent().append('<div class="invalid-feedback">' + (Array.isArray(messages) ? messages[0] : messages) + '</div>');
            }
        });
    }

    // Bind events
    function bindEvents() {
        // Form submission
        $form.off('submit').on('submit', handleSubmit);

        // Offcanvas shown event
        $offcanvas.off('shown.bs.offcanvas').on('shown.bs.offcanvas', function() {
            initializeFlatpickr();
            initializeSelect2();
            $title.focus();
        });

        // Offcanvas hidden event
        $offcanvas.off('hidden.bs.offcanvas').on('hidden.bs.offcanvas', function() {
            resetForm();
        });
    }

    // Public API
    return {
        // Initialize the form handler
        init: function(options = {}) {
            // Merge options with config
            Object.assign(config, options);

            initializeElements();
            bindEvents();
            isInitialized = true;

            return this;
        },

        // Show form for creating new task
        showCreate: function() {
            if (!isInitialized) {
                console.error('TaskFormHandler not initialized');
                return;
            }

            resetForm();
            $offcanvas.offcanvas('show');
            return this;
        },

        // Show form for editing existing task
        showEdit: function(taskId) {
            if (!isInitialized) {
                console.error('TaskFormHandler not initialized');
                return;
            }

            const urls = getUrls();
            const labels = getLabels();

            if (!urls.tasksShow) {
                console.error('TaskFormHandler: Missing tasksShow URL configuration');
                Swal.fire({
                    icon: 'error',
                    title: 'Error',
                    text: 'Configuration error. Please refresh the page and try again.'
                });
                return;
            }

            // Load task data
            $.ajax({
                url: urls.tasksShow.replace('__ID__', taskId),
                method: 'GET',
                success: function(response) {
                    if (response.status === 'success') {
                        resetForm();
                        // Extract task data from response.data.task
                        const taskData = response.data.task || response.data;
                        populateForm(taskData);
                        $offcanvas.offcanvas('show');
                    } else {
                        Swal.fire({
                            icon: 'error',
                            title: 'Error',
                            text: response.data || 'Failed to load task data'
                        });
                    }
                },
                error: function(xhr) {
                    Swal.fire({
                        icon: 'error',
                        title: 'Error',
                        text: xhr.responseJSON?.message || labels.error || 'An error occurred'
                    });
                }
            });

            return this;
        },

        // Check if initialized
        isInitialized: function() {
            return isInitialized;
        },

        // Get form element
        getForm: function() {
            return $form;
        },

        // Get offcanvas element
        getOffcanvas: function() {
            return $offcanvas;
        }
    };
})();
