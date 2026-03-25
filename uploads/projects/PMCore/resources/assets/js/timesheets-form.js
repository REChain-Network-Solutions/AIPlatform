$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Initialize components
    initializeSelect2();
    initializeDatePicker();
    initializeFormHandlers();

    /**
     * Initialize Select2 components
     */
    function initializeSelect2() {
        // User select (for admins/managers)
        $('#user_id').select2({
            placeholder: 'Select User',
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
                    // Check if data is wrapped in a data property or is the array directly
                    const users = data.data || data;
                    return {
                        results: Array.isArray(users) ? users.map(function (user) {
                            return {
                                id: user.id,
                                text: user.text || user.name
                            };
                        }) : []
                    };
                },
                cache: true
            }
        });

        // Project select
        $('#project_id').select2({
            placeholder: 'Select Project',
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
                        results: data.data || []
                    };
                },
                cache: true
            }
        });

        // Task select (dependent on project)
        $('#task_id').select2({
            placeholder: 'Select Task (Optional)',
            allowClear: true
        });
    }

    /**
     * Initialize date picker
     */
    function initializeDatePicker() {
        $('#date').flatpickr({
            dateFormat: 'Y-m-d',
            defaultDate: 'today',
            maxDate: 'today',
            allowInput: true
        });
    }

    /**
     * Initialize form handlers
     */
    function initializeFormHandlers() {
        // Project change handler
        $('#project_id').on('change', function () {
            const projectId = $(this).val();
            
            if (projectId) {
                loadProjectTasks(projectId);
                loadProjectRates(projectId);
            } else {
                // Clear tasks
                $('#task_id').empty().append('<option value="">Select Task (Optional)</option>').trigger('change');
                // Hide rates section
                $('#ratesSection').hide();
            }
        });

        // Billable checkbox handler
        $('#is_billable').on('change', function () {
            if ($(this).is(':checked')) {
                $('#ratesSection').show();
            } else {
                $('#ratesSection').hide();
                $('#billing_rate').val('');
                $('#cost_rate').val('');
            }
        });

        // Form submission
        $('#timesheetForm').on('submit', function (e) {
            e.preventDefault();
            submitForm();
        });

        // Trigger initial billable check
        $('#is_billable').trigger('change');
    }

    /**
     * Load tasks for selected project
     */
    function loadProjectTasks(projectId) {
        const $taskSelect = $('#task_id');
        
        // Show loading state
        $taskSelect.prop('disabled', true);
        $taskSelect.empty().append('<option value="">' + pageData.labels.loadingTasks + '</option>');

        $.ajax({
            url: pageData.urls.projectTasksUrl.replace('__PROJECT_ID__', projectId),
            method: 'GET',
            success: function (response) {
                if (response.status === 'success') {
                    $taskSelect.empty().append('<option value="">Select Task (Optional)</option>');
                    
                    if (response.data.tasks && response.data.tasks.length > 0) {
                        response.data.tasks.forEach(function (task) {
                            $taskSelect.append(new Option(task.text, task.id));
                        });
                    } else {
                        $taskSelect.append('<option value="" disabled>' + pageData.labels.noTasksFound + '</option>');
                    }
                    
                    $taskSelect.prop('disabled', false);
                }
            },
            error: function () {
                $taskSelect.empty().append('<option value="">Select Task (Optional)</option>');
                $taskSelect.prop('disabled', false);
            }
        });
    }

    /**
     * Load project rates (placeholder for future implementation)
     */
    function loadProjectRates(projectId) {
        // This could be implemented to fetch default rates from the project
        // For now, just show the rates section if billable
        if ($('#is_billable').is(':checked')) {
            $('#ratesSection').show();
        }
    }

    /**
     * Submit the form
     */
    function submitForm() {
        const formData = new FormData($('#timesheetForm')[0]);
        
        // Fix checkbox value
        const isBillable = $('#is_billable').is(':checked');
        formData.delete('is_billable');
        formData.append('is_billable', isBillable ? '1' : '0');

        // Clear previous errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');

        $.ajax({
            url: pageData.urls.storeUrl,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response.status === 'success') {
                    // Show success message
                    Swal.fire({
                        icon: 'success',
                        title: 'Success!',
                        text: pageData.labels.createSuccess,
                        showConfirmButton: true,
                        confirmButtonText: 'OK',
                        timer: 3000,
                        timerProgressBar: true
                    }).then(() => {
                        // Redirect to index
                        window.location.href = pageData.urls.indexUrl;
                    });
                } else {
                    handleFormErrors(response.errors || {});
                }
            },
            error: function (xhr) {
                if (xhr.status === 422) {
                    const errors = xhr.responseJSON.errors || {};
                    handleFormErrors(errors);
                } else {
                    Swal.fire({
                        icon: 'error',
                        title: 'Error!',
                        text: pageData.labels.error,
                        showConfirmButton: true
                    });
                }
            }
        });
    }

    /**
     * Handle form validation errors
     */
    function handleFormErrors(errors) {
        Object.keys(errors).forEach(function (field) {
            const input = $(`#${field}`);
            const feedback = input.siblings('.invalid-feedback');
            
            if (input.length) {
                input.addClass('is-invalid');
                feedback.text(errors[field][0]);
            }
        });

        // Show error notification
        Swal.fire({
            icon: 'error',
            title: 'Validation Error!',
            text: 'Please correct the errors below',
            showConfirmButton: true
        });
    }
});