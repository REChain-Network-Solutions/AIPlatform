$(function () {
    // Initialize DataTable
    const dt = $('#auditLogsTable').DataTable({
        processing: true,
        serverSide: true,
        ajax: {
            url: pageData.urls.datatable,
            data: function (d) {
                d.auditable_type = $('#filter_auditable_type').val();
                d.event = $('#filter_event').val();
                d.user_id = $('#filter_user_id').val();
                d.date_from = $('#filter_date_from').val();
                d.date_to = $('#filter_date_to').val();
            }
        },
        columns: [
            {data: 'user', name: 'user_id'},
            {data: 'auditable', name: 'auditable_type'},
            {data: 'event', name: 'event'},
            {data: 'ip_address', name: 'ip_address'},
            {data: 'created_at', name: 'created_at'},
            {data: 'actions', name: 'actions', orderable: false, searchable: false}
        ],
        order: [[4, 'desc']],
        dom: '<"row"<"col-12 col-md-6"l><"col-12 col-md-6"f>><"table-responsive"t><"row"<"col-12 col-md-6"i><"col-12 col-md-6"p>>',
        language: {
            paginate: {
                next: '<i class="bx bx-chevron-right bx-18px"></i>',
                previous: '<i class="bx bx-chevron-left bx-18px"></i>'
            }
        }
    });

    // Initialize Select2
    $('.select2').select2({
        placeholder: $(this).data('placeholder'),
        allowClear: true
    });

    // Initialize date range picker
    $('#filter_date_range').flatpickr({
        mode: 'range',
        dateFormat: 'Y-m-d',
        onChange: function(selectedDates, dateStr, instance) {
            if (selectedDates.length === 2) {
                $('#filter_date_from').val(formatDate(selectedDates[0]));
                $('#filter_date_to').val(formatDate(selectedDates[1]));
            }
        }
    });

    // Load filters
    loadFilters();

    // Load statistics
    loadStatistics();

    // Initialize charts
    initializeCharts();
});

function formatDate(date) {
    return date.getFullYear() + '-' + 
           String(date.getMonth() + 1).padStart(2, '0') + '-' + 
           String(date.getDate()).padStart(2, '0');
}

function loadFilters() {
    $.get(pageData.urls.filters, function(response) {
        if (response.status === 'success') {
            // Load auditable types
            const auditableTypes = response.data.auditableTypes;
            $('#filter_auditable_type').empty().append('<option value="">' + pageData.labels.allModels + '</option>');
            auditableTypes.forEach(type => {
                $('#filter_auditable_type').append(`<option value="${type.value}">${type.label}</option>`);
            });

            // Load events
            const events = response.data.events;
            $('#filter_event').empty().append('<option value="">' + pageData.labels.allEvents + '</option>');
            events.forEach(event => {
                $('#filter_event').append(`<option value="${event.value}">${event.label}</option>`);
            });

            // Load users via Select2 AJAX
            $('#filter_user_id').select2({
                ajax: {
                    url: '/api/users/search',
                    dataType: 'json',
                    delay: 250,
                    data: function (params) {
                        return {
                            search: params.term,
                            page: params.page
                        };
                    },
                    processResults: function (data, params) {
                        params.page = params.page || 1;
                        return {
                            results: data.data.map(user => ({
                                id: user.id,
                                text: user.full_name
                            })),
                            pagination: {
                                more: data.current_page < data.last_page
                            }
                        };
                    }
                },
                placeholder: pageData.labels.allUsers,
                allowClear: true
            });
        }
    });
}

function loadStatistics() {
    $.get(pageData.urls.statistics, function(response) {
        if (response.status === 'success') {
            const data = response.data;
            
            // Update cards
            $('#totalAudits').text(data.totalAudits.toLocaleString());
            $('#createdCount').text((data.auditsByEvent.created || 0).toLocaleString());
            $('#updatedCount').text((data.auditsByEvent.updated || 0).toLocaleString());
            $('#deletedCount').text((data.auditsByEvent.deleted || 0).toLocaleString());

            // Update charts
            updateDailyTrendChart(data.dailyTrend);
            updateActiveUsersChart(data.activeUsers);
        }
    });
}

let dailyTrendChart, activeUsersChart;

function initializeCharts() {
    // Daily Trend Chart
    const dailyTrendOptions = {
        series: [{
            name: pageData.labels.auditLogs,
            data: []
        }],
        chart: {
            height: 350,
            type: 'area',
            toolbar: {
                show: false
            }
        },
        dataLabels: {
            enabled: false
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        xaxis: {
            type: 'datetime'
        },
        yaxis: {
            title: {
                text: pageData.labels.numberOfLogs
            }
        },
        tooltip: {
            x: {
                format: 'dd MMM yyyy'
            }
        }
    };

    dailyTrendChart = new ApexCharts(document.querySelector("#dailyTrendChart"), dailyTrendOptions);
    dailyTrendChart.render();

    // Active Users Chart
    const activeUsersOptions = {
        series: [],
        chart: {
            height: 350,
            type: 'donut'
        },
        labels: [],
        responsive: [{
            breakpoint: 480,
            options: {
                chart: {
                    width: 200
                },
                legend: {
                    position: 'bottom'
                }
            }
        }]
    };

    activeUsersChart = new ApexCharts(document.querySelector("#activeUsersChart"), activeUsersOptions);
    activeUsersChart.render();
}

function updateDailyTrendChart(data) {
    dailyTrendChart.updateSeries([{
        name: pageData.labels.auditLogs,
        data: data
    }]);
}

function updateActiveUsersChart(data) {
    const labels = data.map(item => item.user);
    const series = data.map(item => item.total);
    
    activeUsersChart.updateOptions({
        labels: labels,
        series: series
    });
}

function applyFilters() {
    $('#auditLogsTable').DataTable().ajax.reload();
    loadStatistics();
}

function clearFilters() {
    $('#filterForm')[0].reset();
    $('#filter_user_id').val(null).trigger('change');
    $('.select2').val(null).trigger('change');
    $('#filter_date_range').flatpickr().clear();
    $('#filter_date_from').val('');
    $('#filter_date_to').val('');
    applyFilters();
}

function viewAuditDetails(id) {
    const offcanvas = new bootstrap.Offcanvas(document.getElementById('auditDetailsOffcanvas'));
    
    $('#auditDetailsContent').html('<div class="text-center py-4"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
    offcanvas.show();

    $.get(pageData.urls.show.replace(':id', id), function(response) {
        if (response.status === 'success') {
            $('#auditDetailsContent').html(response.data.html);
        }
    });
}

// Export functions to global scope
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;
window.viewAuditDetails = viewAuditDetails;