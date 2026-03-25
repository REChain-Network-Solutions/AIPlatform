'use strict';

import { Calendar } from '@fullcalendar/core';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import listPlugin from '@fullcalendar/list';
import interactionPlugin from '@fullcalendar/interaction';

$(function () {
    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    let calendar;
    let currentView = 'month';

    // Initialize calendar
    initializeCalendar();
    initializeForm();

    function initializeCalendar() {
        const calendarEl = document.getElementById('resourceCalendar');
        
        // Prepare events from allocations
        const events = [];
        
        pageData.allocations.forEach(function(allocation) {
            // Use native date handling instead of moment
            let endDate;
            if (allocation.end_date) {
                // Add one day for FullCalendar's exclusive end date
                const date = new Date(allocation.end_date);
                date.setDate(date.getDate() + 1);
                endDate = date.toISOString().split('T')[0];
            } else {
                // Set to one year from now if no end date
                const date = new Date();
                date.setFullYear(date.getFullYear() + 1);
                endDate = date.toISOString().split('T')[0];
            }
            
            events.push({
                id: allocation.id,
                title: allocation.project.name + ' (' + allocation.allocation_percentage + '%)',
                start: allocation.start_date,
                end: endDate,
                backgroundColor: getColorForAllocation(allocation),
                borderColor: getColorForAllocation(allocation),
                extendedProps: {
                    allocation: allocation
                }
            });
        });

        // Add capacity indicators
        Object.keys(pageData.capacities).forEach(function(date) {
            const capacity = pageData.capacities[date];
            if (!capacity.is_working_day) {
                events.push({
                    title: 'Non-working day',
                    start: date,
                    display: 'background',
                    backgroundColor: '#e0e0e0'
                });
            }
        });

        // Use the imported Calendar with plugins
        calendar = new Calendar(calendarEl, {
            plugins: [dayGridPlugin, timeGridPlugin, listPlugin, interactionPlugin],
            initialView: 'dayGridMonth',
            headerToolbar: {
                left: 'prev,next today',
                center: 'title',
                right: 'dayGridMonth,timeGridWeek,listMonth'
            },
            events: events,
            eventClick: function(info) {
                if (info.event.extendedProps.allocation) {
                    showAllocationDetails(info.event.extendedProps.allocation);
                }
            },
            height: 'auto',
            editable: false,
            dayMaxEvents: true,
            navLinks: true,
            selectable: true,
            select: function(info) {
                // Pre-fill date when selecting on calendar
                $('#start_date').val(info.startStr);
                if (info.endStr && info.startStr !== info.endStr) {
                    // Adjust end date for exclusive end in FullCalendar
                    const endDate = new Date(info.endStr);
                    endDate.setDate(endDate.getDate() - 1);
                    $('#end_date').val(endDate.toISOString().split('T')[0]);
                }
                showAllocationForm();
            }
        });

        calendar.render();
    }

    function getColorForAllocation(allocation) {
        if (allocation.status === 'cancelled') {
            return '#6c757d'; // gray
        } else if (allocation.allocation_percentage >= 100) {
            return '#dc3545'; // red
        } else if (allocation.allocation_percentage >= 80) {
            return '#ffc107'; // yellow
        } else {
            return '#0d6efd'; // blue
        }
    }

    function initializeForm() {
        // Initialize date pickers
        $('#start_date').flatpickr({
            dateFormat: 'Y-m-d',
            minDate: 'today'
        });

        $('#end_date').flatpickr({
            dateFormat: 'Y-m-d',
            minDate: 'today'
        });

        // Initialize project select
        $('#project_id').select2({
            dropdownParent: $('#allocationModal'),
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
        });

        // Initialize task select
        $('#task_id').select2({
            dropdownParent: $('#allocationModal'),
            placeholder: 'Select Task'
        });

        // Allocation type change handler
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
        $('#allocationForm').on('submit', function(e) {
            e.preventDefault();
            submitAllocation();
        });
    }

    function loadProjectTasks() {
        const projectId = $('#project_id').val();
        if (!projectId) return;

        $('#task_id').empty().append('<option value="">Loading...</option>');
        
        $.ajax({
            url: '/pmcore/projects/' + projectId + '/tasks',
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

    window.showAllocationForm = function() {
        $('#allocationForm')[0].reset();
        $('#allocation_type').val('project').trigger('change'); // Reset to project type
        $('#allocationModal').modal('show');
    };

    window.changeView = function(view) {
        let viewName;
        if (view === 'month') {
            viewName = 'dayGridMonth';
        } else if (view === 'week') {
            viewName = 'timeGridWeek';
        } else if (view === 'list') {
            viewName = 'listMonth';
        }
        
        if (viewName && calendar) {
            calendar.changeView(viewName);
        }
    };

    window.deleteAllocation = function(id) {
        Swal.fire({
            title: 'Are you sure?',
            text: "This allocation will be permanently deleted!",
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Yes, delete it!',
            cancelButtonText: 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.deleteAllocation.replace(':id', id),
                    method: 'DELETE',
                    success: function(response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: 'Deleted!',
                                text: 'The allocation has been deleted.',
                                showConfirmButton: true,
                                timer: 3000,
                                timerProgressBar: true
                            }).then(() => {
                                location.reload();
                            });
                        }
                    },
                    error: function() {
                        Swal.fire({
                            icon: 'error',
                            title: 'Error!',
                            text: 'Failed to delete allocation',
                            showConfirmButton: true
                        });
                    }
                });
            }
        });
    };

    function showAllocationDetails(allocation) {
        // Format dates without moment
        const startDate = new Date(allocation.start_date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        const endDate = allocation.end_date ? new Date(allocation.end_date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' }) : 'Ongoing';
        
        const content = `
            <div>
                <p><strong>Project:</strong> ${allocation.project.name}</p>
                <p><strong>Allocation:</strong> ${allocation.allocation_percentage}%</p>
                <p><strong>Hours/Day:</strong> ${allocation.hours_per_day}</p>
                <p><strong>Period:</strong> ${startDate} - ${endDate}</p>
                <p><strong>Status:</strong> ${allocation.status}</p>
                ${allocation.notes ? `<p><strong>Notes:</strong> ${allocation.notes}</p>` : ''}
            </div>
        `;

        Swal.fire({
            title: 'Allocation Details',
            html: content,
            showConfirmButton: true,
            confirmButtonText: 'Close'
        });
    }

    function submitAllocation() {
        const formData = new FormData($('#allocationForm')[0]);
        
        // Fix checkbox value
        formData.delete('is_billable');
        formData.append('is_billable', $('#is_billable').is(':checked') ? '1' : '0');

        // Clear previous errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');

        $.ajax({
            url: pageData.urls.createAllocation,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'success') {
                    $('#allocationModal').modal('hide');
                    
                    Swal.fire({
                        icon: 'success',
                        title: 'Success!',
                        text: response.data.message,
                        showConfirmButton: true,
                        timer: 3000,
                        timerProgressBar: true
                    }).then(() => {
                        location.reload();
                    });
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
                        title: 'Error!',
                        text: 'An error occurred',
                        showConfirmButton: true
                    });
                }
            }
        });
    }
});