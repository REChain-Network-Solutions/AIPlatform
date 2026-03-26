<?php

/*
 * Inject Css file for logtracker module
 */
hooks()->add_action('app_admin_head', 'logtracker_load_css');
function logtracker_load_css()
{
    if (get_instance()->app_modules->is_active(LOGTRACKER_MODULE)) {
        echo '<link href="' . module_dir_url(LOGTRACKER_MODULE, 'assets/css/logtracker.css') . '?v=' . get_instance()->app_scripts->core_version() . '" rel="stylesheet" type="text/css"></link>';
    }
}

/*
 * Inject Javascript file for logtracker module
 */
hooks()->add_action('app_admin_footer', 'logtracker_load_js');
function logtracker_load_js()
{
    if (get_instance()->app_modules->is_active(LOGTRACKER_MODULE)) {
        echo '<script src="' . module_dir_url(LOGTRACKER_MODULE, 'assets/js/logtracker.js') . '?v=' . get_instance()->app_scripts->core_version() . '"></script>';
    }
}

/*
 * Inject Javascript file for logtracker module
 */
hooks()->add_action('app_admin_footer', function () {
    if (get_instance()->app_modules->is_active('logtracker')) {
        echo '<script src="'.module_dir_url('logtracker', 'assets/js/logtracker.js').'?v='.get_instance()->app_scripts->core_version().'"></script>';
    }
});


hooks()->add_action('pre_activate_module', LOGTRACKER_MODULE . '_sidecheck');
function logtracker_sidecheck($module_name)
{
}

hooks()->add_action('pre_deactivate_module', LOGTRACKER_MODULE . '_deregister');
function logtracker_deregister($module_name)
{
}