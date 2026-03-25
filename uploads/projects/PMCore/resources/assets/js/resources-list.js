$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // DataTable
    let resourcesTable = $('#resourcesTable').DataTable({
        processing: true,
        serverSide: true,
        ajax: {
            url: pageData.urls.datatable
        },
        columns: [
            { data: 'user', name: 'name', orderable: false },
            { data: 'role', name: 'role', orderable: false },
            { data: 'current_allocation', name: 'current_allocation', orderable: false },
            { data: 'availability', name: 'availability', orderable: false },
            { data: 'actions', name: 'actions', orderable: false, searchable: false }
        ],
        order: [[0, 'asc']],
        responsive: true
    });

    // Initialize components
    initializeDatePickers();
    initializeSelect2();
    initializeFormHandlers();

    // Allocate resource function
    window.allocateResource = function(userId, userName) {
        $('#resource_user_id').val(userId);
        $('#resource_name').text(userName || 'User #' + userId);
        
        // Reset form
        $('#allocateResourceForm')[0].reset();
        $('#resource_user_id').val(userId); // Reset after form reset
        $('#allocation_percentage').val(100);
        $('#hours_per_day').val(8);
        $('#availability_preview').empty();
        
        // Show offcanvas
        const offcanvas = new bootstrap.Offcanvas(document.getElementById('allocateResourceOffcanvas'));
        offcanvas.show();
        
        // Check availability when dates change
        checkAvailability();
    };

    // Initialize date pickers
    function initializeDatePickers() {
        $('#start_date').flatpickr({
            dateFormat: 'Y-m-d',
            minDate: 'today',
            onChange: function(selectedDates) {
                if (selectedDates.length > 0) {
                    $('#end_date').flatpickr().set('minDate', selectedDates[0]);
                    checkAvailability();
                }
            }
        });

        $('#end_date').flatpickr({
            dateFormat: 'Y-m-d',
            minDate: 'today',
            onChange: function() {
                checkAvailability();
            }
        });
    }

    // Initialize Select2
    function initializeSelect2() {
        // Project select
        $('#project_id').select2({
            dropdownParent: $('#allocateResourceOffcanvas'),
            placeholder: 'Select Project',
            ajax: {
                url: pageData.urls.projectSearch || '/projects/search',
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
            dropdownParent: $('#allocateResourceOffcanvas'),
            placeholder: 'Select Task'
        });
    }

    // Initialize form handlers
    function initializeFormHandlers() {
        // Allocation type change
        $('#allocation_type').on('change', function() {
            const type = $(this).val();
            $('#phase_section').toggle(type === 'phase');
            $('#task_section').toggle(type === 'task');
            
            if (type === 'phase') {
                $('#phase').prop('required', true);
                $('#task_id').prop('required', false);
            } else if (type === 'task') {
                $('#phase').prop('required', false);
                $('#task_id').prop('required', true);
                loadProjectTasks();
            } else {
                $('#phase').prop('required', false);
                $('#task_id').prop('required', false);
            }
        });

        // Project change - load tasks
        $('#project_id').on('change', function() {
            if ($('#allocation_type').val() === 'task') {
                loadProjectTasks();
            }
        });

        // Form submission
        $('#allocateResourceForm').on('submit', function(e) {
            e.preventDefault();
            submitAllocation();
        });
    }

    // Load project tasks
    function loadProjectTasks() {
        const projectId = $('#project_id').val();
        if (!projectId) return;

        $('#task_id').empty().append('<option value="">Loading...</option>');
        
        $.ajax({
            url: '/projects/' + projectId + '/tasks',
            method: 'GET',
            success: function(response) {
                $('#task_id').empty().append('<option value="">Select Task</option>');
                if (response.data && response.data.tasks) {
                    response.data.tasks.forEach(function(task) {
                        $('#task_id').append(new Option(task.title, task.id));
                    });
                }
            }
        });
    }

    // Check availability
    function checkAvailability() {
        const userId = $('#resource_user_id').val();
        const startDate = $('#start_date').val();
        const endDate = $('#end_date').val() || startDate;
        
        if (!userId || !startDate) return;

        $.ajax({
            url: pageData.urls.availability,
            method: 'POST',
            data: {
                user_id: userId,
                start_date: startDate,
                end_date: endDate
            },
            success: function(response) {
                if (response.status === 'success') {
                    displayAvailability(response.data.availability);
                }
            }
        });
    }

    // Display availability preview
    function displayAvailability(availability) {
        let html = '<div class="alert alert-info"><strong>Availability Preview:</strong><br>';
        let hasConflicts = false;
        
        availability.forEach(function(day) {
            if (day.is_overallocated) {
                hasConflicts = true;
                html += '<span class="text-danger">' + day.date + ': ' + day.allocation_percentage + '% allocated</span><br>';
            }
        });
        
        if (!hasConflicts) {
            html += '<span class="text-success">Resource is available for the selected period</span>';
        }
        
        html += '</div>';
        $('#availability_preview').html(html);
    }

    // Submit allocation
    function submitAllocation() {
        const formData = new FormData($('#allocateResourceForm')[0]);
        
        // Fix checkbox values
        formData.delete('is_billable');
        formData.append('is_billable', $('#is_billable').is(':checked') ? '1' : '0');
        formData.delete('is_confirmed');
        formData.append('is_confirmed', $('#is_confirmed').is(':checked') ? '1' : '0');

        // Clear previous errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');

        $.ajax({
            url: pageData.urls.create.replace('/create', ''),
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'success') {
                    // Hide offcanvas
                    const offcanvas = bootstrap.Offcanvas.getInstance(document.getElementById('allocateResourceOffcanvas'));
                    offcanvas.hide();
                    
                    // Show success message
                    Swal.fire({
                        icon: 'success',
                        title: pageData.labels.success,
                        text: response.data.message,
                        showConfirmButton: true,
                        timer: 3000,
                        timerProgressBar: true
                    });
                    
                    // Reload table
                    resourcesTable.ajax.reload();
                    
                    // Show conflicts warning if any
                    if (response.data.conflicts && response.data.conflicts.length > 0) {
                        setTimeout(function() {
                            Swal.fire({
                                icon: 'warning',
                                title: 'Overallocation Warning',
                                text: 'The resource is overallocated in some periods. Please review the schedule.',
                                showConfirmButton: true
                            });
                        }, 3500);
                    }
                }
            },
            error: function(xhr) {
                if (xhr.status === 422) {
                    const errors = xhr.responseJSON.errors || {};
                    Object.keys(errors).forEach(function(field) {
                        const input = $(`#${field}`);
                        const feedback = input.siblings('.invalid-feedback');
                        
                        if (input.length) {
                            input.addClass('is-invalid');
                            feedback.text(errors[field][0]);
                        }
                    });
                } else {
                    Swal.fire({
                        icon: 'error',
                        title: pageData.labels.error,
                        text: 'An error occurred',
                        showConfirmButton: true
                    });
                }
            }
        });
    }
});