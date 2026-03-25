$(function () {
    'use strict';

    // Chart colors
    const chartColors = {
        primary: '#007bff',
        success: '#28a745',
        info: '#17a2b8',
        warning: '#ffc107',
        danger: '#dc3545',
        secondary: '#6c757d'
    };

    // Project Status Distribution Chart
    if (document.getElementById('projectStatusChart')) {
        const statusChart = new ApexCharts(document.getElementById('projectStatusChart'), {
            chart: {
                type: 'donut',
                height: 300
            },
            series: dashboardData.chartData.status_distribution.data,
            labels: dashboardData.chartData.status_distribution.labels,
            colors: dashboardData.chartData.status_distribution.colors,
            legend: {
                position: 'bottom'
            },
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
        });
        statusChart.render();
    }

    // Monthly Project Creation Chart
    if (document.getElementById('monthlyCreationChart')) {
        const monthlyChart = new ApexCharts(document.getElementById('monthlyCreationChart'), {
            chart: {
                type: 'bar',
                height: 300
            },
            series: [{
                name: 'Projects Created',
                data: dashboardData.chartData.monthly_creation.data
            }],
            xaxis: {
                categories: dashboardData.chartData.monthly_creation.labels
            },
            colors: [chartColors.primary],
            plotOptions: {
                bar: {
                    borderRadius: 4,
                    columnWidth: '60%'
                }
            },
            dataLabels: {
                enabled: false
            },
            grid: {
                borderColor: '#e7e7e7',
                strokeDashArray: 5
            }
        });
        monthlyChart.render();
    }

    // Project Completion Trend Chart
    if (document.getElementById('completionTrendChart')) {
        const completionChart = new ApexCharts(document.getElementById('completionTrendChart'), {
            chart: {
                type: 'line',
                height: 300
            },
            series: [{
                name: 'Projects Completed',
                data: dashboardData.chartData.completion_trend.data
            }],
            xaxis: {
                categories: dashboardData.chartData.completion_trend.labels
            },
            colors: [chartColors.success],
            stroke: {
                curve: 'smooth',
                width: 3
            },
            markers: {
                size: 6
            },
            grid: {
                borderColor: '#e7e7e7',
                strokeDashArray: 5
            }
        });
        completionChart.render();
    }
});
