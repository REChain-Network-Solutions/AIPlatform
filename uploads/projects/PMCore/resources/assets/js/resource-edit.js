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
    initializeFormHandlers();

    // Initialize date pickers
    function initializeDatePickers() {
        $('#start_date').flatpickr({
            dateFormat: 'Y-m-d',
            minDate: 'today'
        });

        $('#end_date').flatpickr({
            dateFormat: 'Y-m-d',
            minDate: 'today'
        });
    }

    // Initialize form handlers
    function initializeFormHandlers() {
        $('#editAllocationForm').on('submit', function(e) {
            e.preventDefault();
            updateAllocation();
        });

        // Status change affects is_confirmed
        $('#status').on('change', function() {
            if ($(this).val() === 'active') {
                $('#is_confirmed').prop('checked', true);
            }
        });
    }

    // Update allocation
    function updateAllocation() {
        const formData = new FormData($('#editAllocationForm')[0]);
        
        // Fix checkbox values
        formData.delete('is_billable');
        formData.append('is_billable', $('#is_billable').is(':checked') ? '1' : '0');
        formData.delete('is_confirmed');
        formData.append('is_confirmed', $('#is_confirmed').is(':checked') ? '1' : '0');

        // Add PUT method
        formData.append('_method', 'PUT');

        // Clear previous errors
        $('.is-invalid').removeClass('is-invalid');
        $('.invalid-feedback').text('');

        $.ajax({
            url: pageData.urls.update,
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
                        window.location.href = pageData.urls.index;
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
                        text: 'An error occurred while updating the allocation',
                        showConfirmButton: true
                    });
                }
            }
        });
    }
});