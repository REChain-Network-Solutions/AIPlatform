$(function () {
    'use strict';

    // CSRF token setup
    $.ajaxSetup({
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

    let currentView = 'utilization';
    let capacityChart = null;

    // Initialize
    initializeDatePickers();
    initializeViewToggle();
    renderChart('utilization');

    // Initialize date pickers
    function initializeDatePickers() {
        $('#start_date').flatpickr({
            dateFormat: 'Y-m-d',
            onChange: function() {
                // Auto-submit form when date changes
                $('#capacityFilterForm').submit();
            }
        });

        $('#end_date').flatpickr({
            dateFormat: 'Y-m-d',
            onChange: function() {
                // Auto-submit form when date changes
                $('#capacityFilterForm').submit();
            }
        });
    }

    // Initialize view toggle buttons
    function initializeViewToggle() {
        $('.btn-group button[data-view]').on('click', function() {
            const $btn = $(this);
            const view = $btn.data('view');
            
            // Update active state
            $('.btn-group button[data-view]').removeClass('active');
            $btn.addClass('active');
            
            // Render new chart
            currentView = view;
            loadChartData(view);
        });
    }

    // Load chart data via AJAX
    function loadChartData(viewType) {
        const filters = {
            start_date: $('#start_date').val(),
            end_date: $('#end_date').val(),
            department_id: $('#department_id').val(),
            view_type: viewType
        };

        $.ajax({
            url: pageData.urls.capacityData,
            method: 'GET',
            data: filters,
            success: function(response) {
                if (response.status === 'success') {
                    renderChart(viewType, response.data);
                }
            },
            error: function() {
                Swal.fire({
                    icon: 'error',
                    title: 'Error',
                    text: 'Failed to load capacity data',
                    showConfirmButton: true
                });
            }
        });
    }

    // Render chart based on view type
    function renderChart(viewType, data) {
        // Destroy existing chart
        if (capacityChart) {
            capacityChart.destroy();
        }

        const chartEl = document.querySelector('#capacityChart');
        
        switch (viewType) {
            case 'utilization':
                renderUtilizationChart(chartEl, data);
                break;
            case 'forecast':
                renderForecastChart(chartEl, data);
                break;
            case 'heatmap':
                renderHeatmapChart(chartEl, data);
                break;
        }
    }

    // Render utilization bar chart
    function renderUtilizationChart(chartEl, data) {
        // Use initial data if no AJAX data provided
        if (!data) {
            data = {
                categories: pageData.capacityData.resource_breakdown.map(item => item.resource.name),
                series: [
                    {
                        name: pageData.labels.allocated,
                        data: pageData.capacityData.resource_breakdown.map(item => 
                            Math.round(Math.min(item.utilization_percentage, 100))
                        )
                    },
                    {
                        name: pageData.labels.available,
                        data: pageData.capacityData.resource_breakdown.map(item => 
                            Math.round(Math.max(0, 100 - item.utilization_percentage))
                        )
                    }
                ]
            };
        }

        const options = {
            chart: {
                type: 'bar',
                height: 400,
                stacked: true,
                toolbar: {
                    show: true,
                    offsetY: -10
                }
            },
            plotOptions: {
                bar: {
                    horizontal: true,
                    barHeight: '70%'
                }
            },
            series: data.series,
            xaxis: {
                categories: data.categories,
                max: 100,
                labels: {
                    formatter: function(val) {
                        return val + '%';
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        fontSize: '12px'
                    }
                }
            },
            colors: ['#435971', '#71dd37'],
            legend: {
                position: 'top',
                horizontalAlign: 'left',
                offsetX: 0,
                offsetY: 0,
                markers: {
                    width: 12,
                    height: 12
                }
            },
            fill: {
                opacity: 1
            },
            dataLabels: {
                enabled: true,
                formatter: function(val) {
                    return val > 0 ? val + '%' : '';
                }
            },
            tooltip: {
                y: {
                    formatter: function(val) {
                        return val + '%';
                    }
                }
            },
            annotations: {
                xaxis: [
                    {
                        x: 100,
                        borderColor: '#ff3e1d',
                        strokeDashArray: 5,
                        label: {
                            borderColor: '#ff3e1d',
                            style: {
                                color: '#fff',
                                background: '#ff3e1d'
                            },
                            text: 'Over capacity'
                        }
                    }
                ]
            }
        };

        capacityChart = new ApexCharts(chartEl, options);
        capacityChart.render();
    }

    // Render forecast line chart
    function renderForecastChart(chartEl, data) {
        if (!data) {
            // Generate dummy forecast data
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
            data = {
                categories: months,
                series: [
                    {
                        name: 'Current Allocation',
                        data: [65, 70, 68, 72, 75, 78]
                    },
                    {
                        name: 'Planned Allocation',
                        data: [70, 75, 80, 85, 88, 92]
                    },
                    {
                        name: 'Forecasted Demand',
                        data: [72, 77, 82, 88, 93, 98]
                    }
                ]
            };
        }

        const options = {
            chart: {
                type: 'line',
                height: 400,
                toolbar: {
                    show: true
                }
            },
            series: data.series,
            xaxis: {
                categories: data.categories
            },
            yaxis: {
                title: {
                    text: 'Utilization %'
                },
                max: 110,
                labels: {
                    formatter: function(val) {
                        return val + '%';
                    }
                }
            },
            stroke: {
                curve: 'smooth',
                width: 3
            },
            markers: {
                size: 5
            },
            colors: ['#435971', '#03c3ec', '#ff3e1d'],
            legend: {
                position: 'top'
            },
            annotations: {
                yaxis: [
                    {
                        y: 100,
                        borderColor: '#ff3e1d',
                        strokeDashArray: 5,
                        label: {
                            borderColor: '#ff3e1d',
                            style: {
                                color: '#fff',
                                background: '#ff3e1d'
                            },
                            text: 'Max Capacity'
                        }
                    }
                ]
            },
            tooltip: {
                shared: true,
                intersect: false,
                y: {
                    formatter: function(val) {
                        return val + '%';
                    }
                }
            }
        };

        capacityChart = new ApexCharts(chartEl, options);
        capacityChart.render();
    }

    // Render heatmap chart
    function renderHeatmapChart(chartEl, data) {
        if (!data) {
            // Generate dummy heatmap data
            const resources = pageData.capacityData.resource_breakdown.slice(0, 10).map(item => item.resource.name);
            const dates = [];
            const startDate = new Date(pageData.startDate);
            const endDate = new Date(pageData.endDate);
            
            for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
                if (d.getDay() !== 0 && d.getDay() !== 6) { // Weekdays only
                    dates.push(d.toISOString().split('T')[0]);
                }
            }
            
            data = {
                resources: resources,
                dates: dates.slice(0, 20), // Limit to 20 days for display
                allocations: resources.map(() => 
                    dates.slice(0, 20).map(() => Math.floor(Math.random() * 100))
                )
            };
        }

        // Transform data for ApexCharts heatmap
        const series = [];
        data.resources.forEach((resource, resourceIndex) => {
            const resourceData = {
                name: resource,
                data: []
            };
            
            data.dates.forEach((date, dateIndex) => {
                resourceData.data.push({
                    x: date,
                    y: data.allocations[resourceIndex][dateIndex]
                });
            });
            
            series.push(resourceData);
        });

        const options = {
            chart: {
                type: 'heatmap',
                height: 400,
                toolbar: {
                    show: true
                }
            },
            series: series,
            dataLabels: {
                enabled: false
            },
            colors: ['#71dd37'],
            xaxis: {
                type: 'category',
                labels: {
                    rotate: -45,
                    rotateAlways: true,
                    formatter: function(val) {
                        const date = new Date(val);
                        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        fontSize: '12px'
                    }
                }
            },
            plotOptions: {
                heatmap: {
                    radius: 2,
                    enableShades: true,
                    shadeIntensity: 0.5,
                    colorScale: {
                        ranges: [
                            {
                                from: 0,
                                to: 70,
                                name: 'Available',
                                color: '#71dd37'
                            },
                            {
                                from: 71,
                                to: 90,
                                name: 'Near Capacity',
                                color: '#ffab00'
                            },
                            {
                                from: 91,
                                to: 100,
                                name: 'At Capacity',
                                color: '#ff3e1d'
                            }
                        ]
                    }
                }
            },
            tooltip: {
                y: {
                    formatter: function(val) {
                        return val + '% allocated';
                    }
                }
            }
        };

        capacityChart = new ApexCharts(chartEl, options);
        capacityChart.render();
    }

    // Department filter change
    $('#department_id').on('change', function() {
        $('#capacityFilterForm').submit();
    });
});