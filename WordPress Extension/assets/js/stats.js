jQuery(document).ready(function($) {
    "use strict";

    const stats_content = document.getElementById('content');
    var data = stats_content.getAttribute('data-data');
    data = JSON.parse( data );

    new Chart(stats_content, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                },
            },
        }
    });

    const stats_generator = document.getElementById('generator');
    var data = stats_generator.getAttribute('data-data');
    data = JSON.parse( data );

    new Chart(stats_generator, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
            }
        }
    });

    const stats_frontend = document.getElementById('frontend_usage');
    var data = stats_frontend.getAttribute('data-data');
    data = JSON.parse( data );

    new Chart(stats_frontend, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
            }
        }
    });

});

