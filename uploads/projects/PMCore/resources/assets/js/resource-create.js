$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Initialize components
    initializeDatePickers();
    initializeSelect2();
    initializeFormHandlers();

    // Check initial availability if we have pre-selected values
    if ((pageData.user || $('#user_id').val()) && $('#start_date').val()) {
        checkAvailability();
    }

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
        // User select
        if (!pageData.user) {
            $('#user_id').select2({
                placeholder: 'Select Resource',
                ajax: {
                    url: pageData.urls.userSearch,
                    dataType: 'json',
                    delay: 250,
                    data: function (params) {
                        return {
                            search: params.term,
                            exclude_roles: ['client', 'customer'],
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
            }).on('change', function() {
                checkAvailability();
            });
        }

        // Project select
        if (!pageData.project) {
            $('#project_id').select2({
                placeholder: 'Select Project',
                ajax: {
                    url: pageData.urls.projectSearch,
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
            }).on('change', function() {
                if ($('#allocation_type').val() === 'task') {
                    loadProjectTasks();
                }
            });
        }

        // Task select
        $('#task_id').select2({
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

        // Form submission
        $('#createAllocationForm').on('submit', function(e) {
            e.preventDefault();
            createAllocation();
        });
    }

    // Load project tasks
    function loadProjectTasks() {
        const projectId = pageData.project?.id || $('#project_id').val();
        if (!projectId) return;

        $('#task_id').empty().append('<option value="">Loading...</option>');
        
        $.ajax({
            url: pageData.urls.projectTasks.replace(':id', projectId),
            method: 'GET',
            success: function(response) {
                $('#task_id').empty().append('<option value="">Select Task</option>');
                if (response.data && response.data.tasks) {
                    response.data.tasks.forEach(function(task) {
                        $('#task_id').append(new Option(task.title, task.id));
                    });
                }
            },
            error: function() {
                $('#task_id').empty().append('<option value="">Failed to load tasks</option>');
            }
        });
    }

    // Check availability
    function checkAvailability() {
        const userId = pageData.user?.id || $('#user_id').val();
        const startDate = $('#start_date').val();
        const endDate = $('#end_date').val() || startDate;
        
        if (!userId || !startDate) {
            $('#availability_info').html(`
                <i class="bx bx-calendar bx-lg text-muted"></i>
                <p class="text-muted mt-2">Select a resource and date range to view availability</p>
            `);
            return;
        }

        $('#availability_info').html(`
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        `);

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
            },
            error: function() {
                $('#availability_info').html(`
                    <i class="bx bx-error-circle bx-lg text-danger"></i>
                    <p class="text-danger mt-2">Failed to check availability</p>
                `);
            }
        });
    }

    // Display availability
    function displayAvailability(availability) {
        let hasConflicts = false;
        let totalAllocated = 0;
        let workingDays = 0;
        
        availability.forEach(function(day) {
            if (day.is_working_day) {
                workingDays++;
                totalAllocated += day.allocation_percentage;
                if (day.is_overallocated) {
                    hasConflicts = true;
                }
            }
        });
        
        const avgAllocation = workingDays > 0 ? Math.round(totalAllocated / workingDays) : 0;
        
        let html = `
            <div class="mb-3">
                <h6>Resource Availability Summary</h6>
            </div>
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>Average Allocation</span>
                    <span class="fw-bold ${avgAllocation > 80 ? 'text-danger' : 'text-success'}">${avgAllocation}%</span>
                </div>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar ${avgAllocation > 100 ? 'bg-danger' : avgAllocation > 80 ? 'bg-warning' : 'bg-success'}" 
                         style="width: ${Math.min(avgAllocation, 100)}%"></div>
                </div>
            </div>
        `;
        
        if (hasConflicts) {
            html += `
                <div class="alert alert-warning">
                    <h6 class="alert-heading">Overallocation Warning</h6>
                    <p class="mb-0">The resource will be overallocated on some days in the selected period.</p>
                </div>
            `;
            
            // Show conflict details
            html += '<div class="mt-3"><h6>Conflict Dates:</h6><ul class="list-unstyled">';
            availability.forEach(function(day) {
                if (day.is_overallocated) {
                    html += `<li class="text-danger"><i class="bx bx-x-circle me-1"></i>${day.date}: ${day.allocation_percentage}% allocated</li>`;
                }
            });
            html += '</ul></div>';
        } else {
            html += `
                <div class="alert alert-success">
                    <h6 class="alert-heading">Available</h6>
                    <p class="mb-0">The resource has capacity available for the selected period.</p>
                </div>
            `;
        }
        
        $('#availability_info').html(html);
        $('#availability_preview').html(hasConflicts ? 
            '<div class="alert alert-warning"><i class="bx bx-error me-1"></i>Resource will be overallocated on some days</div>' : 
            '<div class="alert alert-success"><i class="bx bx-check-circle me-1"></i>Resource is available for allocation</div>'
        );
    }

    // Create allocation
    function createAllocation() {
        const formData = new FormData($('#createAllocationForm')[0]);
        
        // Fix checkbox values
        formData.delete('is_billable');
        formData.append('is_billable', $('#is_billable').is(':checked') ? '1' : '0');
        formData.delete('is_confirmed');
        formData.append('is_confirmed', $('#is_confirmed').is(':checked') ? '1' : '0');

        // Clear previous errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');

        $.ajax({
            url: pageData.urls.store,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'success') {
                    Swal.fire({
                        icon: 'success',
                        title: pageData.labels.success,
                        text: response.data.message,
                        showConfirmButton: true,
                        timer: 3000,
                        timerProgressBar: true
                    }).then(() => {
                        // Redirect based on where we came from
                        if (pageData.project) {
                            window.location.href = '/pmcore/projects/' + pageData.project.id;
                        } else {
                            window.location.href = pageData.urls.index;
                        }
                    });
                    
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
                        text: 'An error occurred while creating the allocation',
                        showConfirmButton: true
                    });
                }
            }
        });
    }
});