$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    // Initialize Flatpickr for date fields
    const initializeFlatpickr = function() {
        if (typeof flatpickr !== 'undefined') {
            $('.flatpickr-date').flatpickr({
                dateFormat: 'Y-m-d',
                allowInput: true
            });
        }
    };

    // Initialize Select2 for user search (members)
    $('#member_user_id').select2({
        dropdownParent: $('#addMemberModal'),
        ajax: {
            url: pageData.urls.userSearch,
            dataType: 'json',
            delay: 250,
            data: function (params) {
                return {
                    search: params.term
                };
            },
            processResults: function (response) {
                console.log('User search results (modal):', response);
                return {
                    results: response.data || response
                };
            },
            cache: true
        },
        placeholder: 'Search for users...',
        minimumInputLength: 1
    });

    // Initialize Select2 for client search (edit modal)
    const initializeClientSelect2 = function() {
        $('#edit_project_client').select2({
            dropdownParent: $('#editProjectModal'),
            ajax: {
                url: pageData.urls.clientSearch,
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        search: params.term
                    };
                },
                processResults: function (response) {
                    console.log('Client search results:', response);
                    return {
                        results: response.data || response
                    };
                },
                cache: true
            },
            placeholder: 'Search for clients...',
            allowClear: true,
            minimumInputLength: 0
        });
    };

    // Initialize Select2 for project manager search (edit modal)
    const initializeProjectManagerSelect2 = function() {
        $('#edit_project_manager').select2({
            dropdownParent: $('#editProjectModal'),
            ajax: {
                url: pageData.urls.userSearch,
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        search: params.term
                    };
                },
                processResults: function (response) {
                    console.log('Project manager search results:', response);
                    return {
                        results: response.data || response
                    };
                },
                cache: true
            },
            placeholder: 'Search for project managers...',
            allowClear: true,
            minimumInputLength: 0
        });
    };

    // Edit Project Button
    $('.edit-project-btn').on('click', function () {
        const projectId = $(this).data('id');
        const project = pageData.project;

        // Populate edit form with all fields
        $('#edit_project_id').val(project.id);
        $('#edit_project_name').val(project.name || '');
        $('#edit_project_code').val(project.code || '');
        $('#edit_project_status').val(project.status || 'planning');
        $('#edit_project_type').val(project.type || 'internal');
        $('#edit_project_priority').val(project.priority || 'medium');
        $('#edit_project_description').val(project.description || '');
        $('#edit_project_start_date').val(project.start_date || '');
        $('#edit_project_end_date').val(project.end_date || '');
        $('#edit_project_budget').val(project.budget || '');
        $('#edit_project_hourly_rate').val(project.hourly_rate || '');
        $('#edit_project_color').val(project.color_code || '#007bff');
        $('#edit_project_billable').prop('checked', project.is_billable == 1);

        // Clear and set client if exists
        $('#edit_project_client').empty();
        if (project.client_id && project.client) {
            const clientOption = new Option(project.client.name, project.client_id, true, true);
            $('#edit_project_client').append(clientOption).trigger('change');
        }

        // Clear and set project manager if exists
        $('#edit_project_manager').empty();
        if (project.project_manager_id && project.project_manager) {
            const managerName = project.project_manager.first_name + ' ' + project.project_manager.last_name;
            const managerOption = new Option(managerName, project.project_manager_id, true, true);
            $('#edit_project_manager').append(managerOption).trigger('change');
        }

        $('#editProjectModal').modal('show');
    });

    // Edit Project Form Submission
    $('#editProjectForm').on('submit', function (e) {
        e.preventDefault();

        const formData = new FormData(this);
        formData.append('_method', 'PUT');

        // Fix checkbox value - convert "on" to boolean
        const isBillableChecked = $('#edit_project_billable').is(':checked');
        formData.delete('is_billable'); // Remove the default "on" value
        formData.append('is_billable', isBillableChecked ? '1' : '0');

        $.ajax({
            url: pageData.urls.projectUpdate,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response.status === 'success') {
                    Swal.fire({
                        icon: 'success',
                        title: pageData.labels.success,
                        text: response.data.message,
                        timer: 2000,
                        showConfirmButton: false
                    }).then(() => {
                        location.reload(); // Reload to show updated data
                    });
                } else {
                    Swal.fire({
                        icon: 'error',
                        title: pageData.labels.error,
                        text: response.data || 'Failed to update project'
                    });
                }
            },
            error: function (xhr) {
                let message = 'Failed to update project';
                if (xhr.status === 422) {
                    const errors = xhr.responseJSON?.data || xhr.responseJSON?.errors;
                    if (errors && typeof errors === 'object') {
                        message = Object.values(errors)[0];
                    }
                }
                Swal.fire({
                    icon: 'error',
                    title: pageData.labels.error,
                    text: message
                });
            }
        });
    });

    // Add Member Form Submission
    $('#addMemberForm').on('submit', function (e) {
        e.preventDefault();

        const formData = new FormData(this);

        $.ajax({
            url: pageData.urls.addMember,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response.status === 'success') {
                    Swal.fire({
                        icon: 'success',
                        title: pageData.labels.success,
                        text: response.data.message,
                        timer: 2000,
                        showConfirmButton: false
                    }).then(() => {
                        location.reload(); // Reload to show new member
                    });
                } else {
                    Swal.fire({
                        icon: 'error',
                        title: pageData.labels.error,
                        text: response.data || 'Failed to add member'
                    });
                }
            },
            error: function (xhr) {
                let message = 'Failed to add member';
                if (xhr.status === 422) {
                    const errors = xhr.responseJSON?.data || xhr.responseJSON?.errors;
                    if (errors && typeof errors === 'object') {
                        message = Object.values(errors)[0];
                    }
                } else if (xhr.status === 400 && xhr.responseJSON) {
                    // Handle 400 errors with custom message
                    message = xhr.responseJSON.data || xhr.responseJSON.message || message;
                } else if (xhr.responseJSON?.data) {
                    // Handle other error responses with data field
                    message = xhr.responseJSON.data;
                }
                Swal.fire({
                    icon: 'error',
                    title: pageData.labels.error,
                    text: message
                });
            }
        });
    });

    // Edit Member Role
    $(document).on('click', '.edit-member-role', function (e) {
        e.preventDefault();
        const memberId = $(this).data('member-id');

        console.log('Edit member role clicked for member ID:', memberId);
        console.log('URL will be:', pageData.urls.getMemberDetails.replace('__MEMBER_ID__', memberId));

        // Show loading state
        Swal.fire({
            title: pageData.labels.loading || 'Loading...',
            allowOutsideClick: false,
            didOpen: () => {
                Swal.showLoading();
            }
        });

        // Fetch member details via AJAX
        $.ajax({
            url: pageData.urls.getMemberDetails.replace('__MEMBER_ID__', memberId),
            method: 'GET',
            success: function (response) {
                console.log('Member details response:', response);
                Swal.close();

                if (response.status === 'success') {
                    const member = response.data.member;
                    console.log('Member data:', member);

                    // Populate the edit modal with fetched data
                    $('#edit_member_id').val(member.id);
                    $('#edit_member_user_name').val(member.user_name);
                    $('#edit_member_role_select').val(member.role);
                    $('#edit_member_hourly_rate_input').val(member.hourly_rate || '');
                    $('#edit_member_allocation_input').val(member.allocation_percentage || '');

                    console.log('Modal populated with:', {
                        id: member.id,
                        user_name: member.user_name,
                        role: member.role,
                        hourly_rate: member.hourly_rate,
                        allocation_percentage: member.allocation_percentage
                    });

                    // Show the modal
                    $('#editMemberRoleModal').modal('show');
                } else {
                    console.error('Error response:', response);
                    Swal.fire({
                        icon: 'error',
                        title: pageData.labels.error,
                        text: response.data || 'Failed to load member details'
                    });
                }
            },
            error: function (xhr) {
                console.error('AJAX error:', xhr);
                Swal.close();

                let message = 'Failed to load member details';
                if (xhr.status === 404) {
                    message = 'Member not found';
                }

                Swal.fire({
                    icon: 'error',
                    title: pageData.labels.error,
                    text: message
                });
            }
        });
    });

    // Edit Member Role Form Submission
    $('#editMemberRoleForm').on('submit', function (e) {
        e.preventDefault();

        const memberId = $('#edit_member_id').val();
        const formData = new FormData(this);

        $.ajax({
            url: pageData.urls.updateMemberRole.replace('__MEMBER_ID__', memberId),
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            headers: {
                'X-HTTP-Method-Override': 'PUT'
            },
            success: function (response) {
                if (response.status === 'success') {
                    Swal.fire({
                        icon: 'success',
                        title: pageData.labels.success,
                        text: response.data.message || 'Member role updated successfully!',
                        timer: 2000,
                        showConfirmButton: false
                    }).then(() => {
                        $('#editMemberRoleModal').modal('hide');
                        location.reload(); // Reload to show updated member info
                    });
                } else {
                    Swal.fire({
                        icon: 'error',
                        title: pageData.labels.error,
                        text: response.data || 'Failed to update member role'
                    });
                }
            },
            error: function (xhr) {
                let message = 'Failed to update member role';
                if (xhr.status === 422) {
                    const errors = xhr.responseJSON?.data || xhr.responseJSON?.errors;
                    if (errors && typeof errors === 'object') {
                        message = Object.values(errors)[0];
                    }
                }
                Swal.fire({
                    icon: 'error',
                    title: pageData.labels.error,
                    text: message
                });
            }
        });
    });

    // Remove Member
    $(document).on('click', '.remove-member', function (e) {
        e.preventDefault();
        const memberId = $(this).data('member-id');

        Swal.fire({
            title: pageData.labels.removeMemberConfirm,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#3085d6',
            confirmButtonText: 'Yes, remove!',
            cancelButtonText: 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.removeMember.replace('__MEMBER_ID__', memberId),
                    method: 'DELETE',
                    success: function (response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: response.data.message,
                                timer: 2000,
                                showConfirmButton: false
                            }).then(() => {
                                location.reload();
                            });
                        } else {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: response.data || 'Failed to remove member'
                            });
                        }
                    },
                    error: function (xhr) {
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: xhr.responseJSON?.data || 'Failed to remove member'
                        });
                    }
                });
            }
        });
    });

    // Delete Project
    $(document).on('click', '.delete-project', function (e) {
        e.preventDefault();

        Swal.fire({
            title: pageData.labels.deleteConfirm,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#3085d6',
            confirmButtonText: 'Yes, delete!',
            cancelButtonText: 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.projectDelete,
                    method: 'DELETE',
                    success: function (response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: response.data.message || pageData.labels.deleteSuccess,
                                timer: 2000,
                                showConfirmButton: false
                            }).then(() => {
                                window.location.href = '/projects'; // Redirect to projects index
                            });
                        } else {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: response.data || 'Failed to delete project'
                            });
                        }
                    },
                    error: function (xhr) {
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: xhr.responseJSON?.data || 'Failed to delete project'
                        });
                    }
                });
            }
        });
    });

    // Reset forms when modals are closed
    $('#addMemberModal').on('hidden.bs.modal', function () {
        $('#addMemberForm')[0].reset();
        $('#member_user_id').val(null).trigger('change');
    });

    // Reinitialize Select2 when modal is shown
    $('#addMemberModal').on('shown.bs.modal', function () {
        // Destroy and reinitialize Select2 to ensure it works properly
        $('#member_user_id').select2('destroy').select2({
            dropdownParent: $('#addMemberModal'),
            ajax: {
                url: pageData.urls.userSearch,
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        search: params.term
                    };
                },
                processResults: function (response) {
                    console.log('User search results:', response);
                    return {
                        results: response.data || response
                    };
                },
                cache: true
            },
            placeholder: 'Search for users...',
            minimumInputLength: 1
        });
    });

    $('#editProjectModal').on('hidden.bs.modal', function () {
        $('#editProjectForm')[0].reset();
        // Destroy Select2 instances to prevent memory leaks
        $('#edit_project_client').select2('destroy');
        $('#edit_project_manager').select2('destroy');
    });

    // Initialize Select2 and Flatpickr when edit modal is shown
    $('#editProjectModal').on('shown.bs.modal', function () {
        initializeFlatpickr();
        initializeClientSelect2();
        initializeProjectManagerSelect2();
    });

    // Initialize Flatpickr on page load
    initializeFlatpickr();

    // Handle duplicate project
    $(document).on('click', '.duplicate-project', function (e) {
        e.preventDefault();
        const projectId = $(this).data('id');

        Swal.fire({
            title: pageData.labels.duplicateConfirm,
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: 'Yes, duplicate it!',
            cancelButtonText: 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.projectDuplicate,
                    method: 'POST',
                    dataType: 'json',
                    success: function (response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: response.data.message || pageData.labels.duplicateSuccess,
                                confirmButtonText: 'OK'
                            }).then(() => {
                                // Redirect to the new duplicated project
                                if (response.data.redirect) {
                                    window.location.href = response.data.redirect;
                                } else {
                                    window.location.reload();
                                }
                            });
                        } else {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: response.data || 'Failed to duplicate project'
                            });
                        }
                    },
                    error: function (xhr) {
                        const errorMessage = xhr.responseJSON?.data || 'Failed to duplicate project. Please try again.';
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: errorMessage
                        });
                    }
                });
            }
        });
    });

    // Handle archive project
    $(document).on('click', '.archive-project', function (e) {
        e.preventDefault();
        const projectId = $(this).data('id');

        Swal.fire({
            title: pageData.labels.archiveConfirm,
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: 'Yes, archive it!',
            cancelButtonText: 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                $.ajax({
                    url: pageData.urls.projectArchive,
                    method: 'POST',
                    dataType: 'json',
                    success: function (response) {
                        if (response.status === 'success') {
                            Swal.fire({
                                icon: 'success',
                                title: pageData.labels.success,
                                text: response.data.message || pageData.labels.archiveSuccess,
                                confirmButtonText: 'OK'
                            }).then(() => {
                                // Reload the page to show updated status
                                window.location.reload();
                            });
                        } else {
                            Swal.fire({
                                icon: 'error',
                                title: pageData.labels.error,
                                text: response.data || 'Failed to archive project'
                            });
                        }
                    },
                    error: function (xhr) {
                        const errorMessage = xhr.responseJSON?.data || 'Failed to archive project. Please try again.';
                        Swal.fire({
                            icon: 'error',
                            title: pageData.labels.error,
                            text: errorMessage
                        });
                    }
                });
            }
        });
    });

    // Reset forms when modals are closed
    $('#addMemberModal').on('hidden.bs.modal', function () {
        $('#addMemberForm')[0].reset();
        $('#member_user_id').val(null).trigger('change');
    });

    $('#editMemberRoleModal').on('hidden.bs.modal', function () {
        $('#editMemberRoleForm')[0].reset();
        $('#edit_member_id').val('');
        $('#edit_member_user_name').val('');
        $('#edit_member_role_select').val('');
        $('#edit_member_hourly_rate_input').val('');
        $('#edit_member_allocation_input').val('');
    });

    // Reinitialize Select2 when modal is shown
    $('#editProjectModal').on('shown.bs.modal', function () {
        initializeFlatpickr();
        initializeClientSelect2();
        initializeProjectManagerSelect2();
    });

    console.log('Project Show Page Initialized');
});
