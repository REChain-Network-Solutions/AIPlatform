$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Initialize Select2 for AJAX dropdowns
    $('.select2-ajax').each(function() {
        const $this = $(this);
        const url = $this.data('ajax--url');
        const placeholder = $this.data('placeholder');
        
        $this.select2({
            placeholder: placeholder,
            allowClear: true,
            ajax: {
                url: url,
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        q: params.term,
                        page: params.page || 1
                    };
                },
                processResults: function (data, params) {
                    params.page = params.page || 1;
                    return {
                        results: data.results,
                        pagination: {
                            more: data.pagination.more
                        }
                    };
                },
                cache: true
            },
            minimumInputLength: 0
        });
    });

    // Initialize Flatpickr for date fields
    $('.flatpickr').flatpickr({
        dateFormat: 'Y-m-d',
        allowInput: true
    });

    // Handle checkbox value on form submit
    $('#projectForm').on('submit', function(e) {
        // Convert checkbox value to 1 or 0
        const isBillable = $('#is_billable').is(':checked');
        if (!isBillable) {
            // Add a hidden input with value 0 if unchecked
            $('<input>').attr({
                type: 'hidden',
                name: 'is_billable',
                value: '0'
            }).appendTo(this);
        }
    });

    // Auto-generate project code from name
    $('#name').on('blur', function() {
        const name = $(this).val();
        const code = $('#code');
        
        // Only generate if code is empty
        if (name && !code.val()) {
            // Generate code from name (e.g., "My Project" => "MP")
            const generatedCode = name
                .split(' ')
                .map(word => word.charAt(0).toUpperCase())
                .join('')
                .substring(0, 10); // Limit to 10 characters
            
            code.val(generatedCode);
        }
    });

    // If editing, pre-populate data
    if (window.pageData && window.pageData.project) {
        const project = window.pageData.project;
        
        // Pre-select client if exists
        if (project.client) {
            const option = new Option(project.client.name, project.client.id, true, true);
            $('#client_id').append(option).trigger('change');
        }
        
        // Pre-select project manager if exists
        if (project.project_manager) {
            const managerName = project.project_manager.first_name + ' ' + project.project_manager.last_name;
            const option = new Option(managerName, project.project_manager.id, true, true);
            $('#project_manager_id').append(option).trigger('change');
        }
    }
});