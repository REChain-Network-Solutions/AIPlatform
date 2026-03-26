<?php

defined('BASEPATH') or exit('No direct script access allowed');

/*
    Module Name: Logtracker
    Description: Swiftly pinpoint errors, Debug logs and Critical information without hassle in your crm system
    Version: 1.0.0
    Requires at least: 3.0.*
    Author: Terreus
    Author URI: https://discord.gg/zDhAUV3Uzt
*/

require_once __DIR__ . '/vendor/autoload.php';

use WpOrg\Requests\Requests as Logtracker_Requests;

/*
 * Define module name
 * Module Name Must be in CAPITAL LETTERS
 */
define('LOGTRACKER_MODULE', 'logtracker');

/*
 * Register activation module hook
 */
register_activation_hook(LOGTRACKER_MODULE, 'logtracker_module_activate_hook');
function logtracker_module_activate_hook()
{
    require_once __DIR__ . '/install.php';
}

/*
 * Register language files, must be registered if the module is using languages
 */
register_language_files(LOGTRACKER_MODULE, [LOGTRACKER_MODULE]);

require_once __DIR__ . '/includes/assets.php';
require_once __DIR__ . '/includes/staff_permissions.php';
require_once __DIR__ . '/includes/sidebar_menu_links.php';

\modules\logtracker\core\Apiinit::ease_of_mind(LOGTRACKER_MODULE);
\modules\logtracker\core\Apiinit::the_da_vinci_code(LOGTRACKER_MODULE);

require_once __DIR__ . '/install.php';

get_instance()->load->helper(LOGTRACKER_MODULE . '/logtracker');

hooks()->add_filter('get_dashboard_widgets', function ($widgets) {
    $new_widgets = [];
    if (has_permission('logtracker', '', 'view')) {
        $new_widgets[] = [
            'path' => LOGTRACKER_MODULE . '/widgets/logtracker-widget',
            'container' => 'top-12',
        ];
    }

    return array_merge($new_widgets, $widgets);
});

hooks()->add_filter('staff_permissions', function ($permissions) {
    $viewGlobalName = _l('permission_view') . '(' . _l('permission_global') . ')';
    $allPermissionsArray = [
        'view' => $viewGlobalName,
        'download' => _l('permission_download'),
        'email' => _l('permission_email'),
        'delete' => _l('permission_delete'),
    ];
    $permissions['logtracker'] = [
        'name' => _l('logtracker'),
        'capabilities' => $allPermissionsArray,
    ];
    return $permissions;
});

hooks()->add_filter('before_settings_updated', function ($data) {
    if (isset ($data['settings']['enviornment_mode'])) {
        $custom_word = $data['settings']['enviornment_mode'] == 'development' ? 'production' : 'development';

        if (file_exists('index.php')) {
            $fileContents = file_get_contents('index.php');

            $searchString = "define('ENVIRONMENT', '" . $custom_word . "');";
            $replacementString = "define('ENVIRONMENT', '" . $data['settings']['enviornment_mode'] . "');";

            $fileContents = str_replace($searchString, $replacementString, $fileContents);

            file_put_contents('index.php', $fileContents);
        }
    }

    return $data;
});

hooks()->add_action('after_email_templates', function () {
    $data['hasPermissionEdit'] = has_permission('edit', 'email_templates');
    $data['logtracker'] = get_instance()->emails_model->get([
        'type' => 'logtracker',
        'language' => 'english',
    ]);
    get_instance()->load->view('logtracker/mail_lists/email_templates_list', $data, false);
});

hooks()->add_filter('other_merge_fields_available_for', function ($available_for) {
    $available_for[] = 'logtracker';

    return $available_for;
});

create_email_template('Error log: [{error_level}] Error Reported at [{error_time}]', '<p>Hello there,</p>
<p>Our <a href="{crm_url}">CRM</a> has detected an error.</p>
<p>Error Time: {error_time}</p>
<p>Error Level: {error_level}</p>
<p>Error Message: {error_message}</p>
<p>Please review the details and take appropriate action as necessary.</p>', 'logtracker', 'Error log: [{error_level}] Error Reported at [{error_time}]', 'error-log-information');

register_merge_fields(LOGTRACKER_MODULE . '/merge_fields/logtracker_merge_fields');
