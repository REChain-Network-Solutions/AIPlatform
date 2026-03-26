<?php

defined('BASEPATH') or exit('No direct script access allowed');

call_user_func('\modules\logtracker\core\Apiinit::the_da_vinci_code', 'logtracker');

// Default environment mode
add_option('enviornment_mode', ENVIRONMENT);

// Default Log level color
add_option('date_color', '#3498db');
add_option('all_color', '#27ae60');
add_option('critical_color', '#ff7a00');
add_option('error_color', '#e74c3c');
add_option('debug_color', '#9b59b6');
add_option('info_color', '#2c3e50');
