<?php

defined('BASEPATH') or exit('No direct script access allowed');

/*
Module Name: Report Builder
Description: Bring your data to life with our customizable report builder
Version: 1.0.0
Requires at least: 2.3.*
Author: GreenTech Solutions
Author URI: https://codecanyon.net/user/greentech_solutions
*/

define('REPORT_BUILDER_MODULE_NAME', 'report_builder');
define('REPORT_BUILDER_MODULE_UPLOAD_FOLDER', module_dir_path(REPORT_BUILDER_MODULE_NAME, 'uploads'));

define('REPORT_BUILDER_OPERATION_ATTACHMENTS_UPLOAD_FOLDER', module_dir_path(REPORT_BUILDER_MODULE_NAME, 'uploads/operations/'));
define('REPORT_BUILDER_PRODUCT_UPLOAD', module_dir_path(REPORT_BUILDER_MODULE_NAME, 'uploads/products/'));


define('REPORT_BUILDER_PRINT_ITEM', 'modules/report_builder/uploads/print_item/');


hooks()->add_action('admin_init', 'report_builder_permissions');
hooks()->add_action('app_admin_head', 'report_builder_add_head_components');
hooks()->add_action('app_admin_footer', 'report_builder_load_js');
hooks()->add_action('app_search', 'report_builder_load_search');
hooks()->add_action('admin_init', 'report_builder_module_init_menu_items');
hooks()->add_action('report_builder_init',REPORT_BUILDER_MODULE_NAME.'_appint');
hooks()->add_action('pre_activate_module', REPORT_BUILDER_MODULE_NAME.'_preactivate');
hooks()->add_action('pre_deactivate_module', REPORT_BUILDER_MODULE_NAME.'_predeactivate');

define('VERSION_REPORT_BUILDER', 100);

/**
* Register activation module hook
*/
register_activation_hook(REPORT_BUILDER_MODULE_NAME, 'report_builder_module_activation_hook');

function report_builder_module_activation_hook()
{
	$CI = &get_instance();
	require_once(__DIR__ . '/install.php');
}


/**
* Register language files, must be registered if the module is using languages
*/
register_language_files(REPORT_BUILDER_MODULE_NAME, [REPORT_BUILDER_MODULE_NAME]);


$CI = & get_instance();
$CI->load->helper(REPORT_BUILDER_MODULE_NAME . '/report_builder');

/**
 * Init goals module menu items in setup in admin_init hook
 * @return null
 */
function report_builder_module_init_menu_items()
{   
	$CI = &get_instance();

	if(has_permission('report_builder','','view') ){
		
		$CI->app_menu->add_sidebar_menu_item('report_builder', [
			'name'     => _l('report_builder_name'),
			'icon'     => 'fa fa-indent', 
			'position' => 10,
		]);
	}

	if(has_permission('report_builder','','view')){
		$CI->app_menu->add_sidebar_children_item('report_builder', [
			'slug'     => 'report_builder_dashboard',
			'name'     => _l('rb_report_management'),
			'icon'     => 'fa fa-list-alt menu-icon',
			'href'     => admin_url('report_builder/report_manage'),
			'position' => 1,
		]);
	}

	if(has_permission('report_builder','','view')){
		$CI->app_menu->add_sidebar_children_item('report_builder', [
			'slug'     => 'report_builder_template',
			'name'     => _l('report_template'),
			'icon'     => 'fa fa-outdent menu-icon',
			'href'     => admin_url('report_builder/template_manage'),
			'position' => 1,
		]);
	}

	if(has_permission('report_builder','','view')){
		$CI->app_menu->add_sidebar_children_item('report_builder', [
			'slug'     => 'report_share',
			'name'     => _l('rb_report_share'),
			'icon'     => 'fa fa-share-alt',
			'href'     => admin_url('report_builder/sharing_manage'),
			'position' => 1,
		]);
	}

	if(has_permission('report_builder','','edit') || has_permission('report_builder','','create')){
		 $CI->app_menu->add_sidebar_children_item('report_builder', [
			'slug'     => 'report_builder_setting', 
			'name'     => _l('rp_settings'),
			'icon'     => 'fa fa-cog menu-icon',
			'href'     => admin_url('report_builder/setting?group=general_setting'),
			'position' => 10,
		]);
	 }


}

	/**
	 * report_builder load js
	 */
	function report_builder_load_js(){    
		$CI = &get_instance();    
		$viewuri = $_SERVER['REQUEST_URI'];

		if (!(strpos($viewuri, '/admin/report_builder/column_manage') === false) || !(strpos($viewuri, '/admin/report_builder/group_manage') === false)|| !(strpos($viewuri, '/admin/report_builder/subtotal_manage') === false)) {   
		   echo '<script src="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/Side_By_Side_Multi_Select_Plugin_jQuery/jquery.multiselect.js') . '"></script>';
		}

		if (!(strpos($viewuri, '/admin/report_builder/column_manage') === false)) {   
			echo '<script src="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/chosen.jquery.js') . '"></script>';
			echo '<script src="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/handsontable-chosen-editor.js') . '"></script>';
			echo '<script src="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/chosen.jquery.js') . '"></script>';
			echo '<script src="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/handsontable-chosen-editor.js') . '"></script>';
		}

	}


	/**
	 * report_builder add head components
	 */
	function report_builder_add_head_components(){    
		$CI = &get_instance();
		$viewuri = $_SERVER['REQUEST_URI'];

		if(!(strpos($viewuri,'admin/report_builder/data_source_manage') === false) || !(strpos($viewuri,'admin/report_builder/column_manage') === false) || !(strpos($viewuri,'admin/report_builder/cell_formatting') === false)|| !(strpos($viewuri,'admin/report_builder/group_manage') === false)|| !(strpos($viewuri,'admin/report_builder/subtotal_manage') === false)|| !(strpos($viewuri,'admin/report_builder/template_setting_manage') === false)){
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/css/reports/report.css') . '?v=' . VERSION_REPORT_BUILDER. '"  rel="stylesheet" type="text/css" />';
		}

		if (!(strpos($viewuri, '/admin/report_builder/column_manage') === false) || !(strpos($viewuri, '/admin/report_builder/group_manage') === false)|| !(strpos($viewuri, '/admin/report_builder/subtotal_manage') === false)) {
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/Side_By_Side_Multi_Select_Plugin_jQuery/jquery.multiselect.css') . '"  rel="stylesheet" type="text/css" />';
		}

		if(!(strpos($viewuri,'admin/report_builder') === false)){
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/css/style.css') . '?v=' . VERSION_REPORT_BUILDER. '"  rel="stylesheet" type="text/css" />';
		}

		if(!(strpos($viewuri,'admin/report_builder/column_manage') === false)){
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/handsontable.full.min.css') . '"  rel="stylesheet" type="text/css" />';
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/chosen.css') . '"  rel="stylesheet" type="text/css" />';
			echo '<script src="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/plugins/handsontable/handsontable.full.min.js') . '"></script>';
		}

		if(!(strpos($viewuri,'admin/report_builder/sharing_manage') === false) || !(strpos($viewuri,'admin/report_builder/report_sharing_setting') === false)){
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/css/sharings/sharing.css') . '?v=' . VERSION_REPORT_BUILDER. '"  rel="stylesheet" type="text/css" />';

		}

		if(!(strpos($viewuri,'admin/report_builder/cell_formatting') === false) || !(strpos($viewuri,'admin/report_builder/data_source_manage') === false) ){
			echo '<link href="' . module_dir_url(REPORT_BUILDER_MODULE_NAME, 'assets/css/templates/cell_formatting.css') . '?v=' . VERSION_REPORT_BUILDER. '"  rel="stylesheet" type="text/css" />';

		}

	}



	/**
	 * report_builder permissions
	 */
	function report_builder_permissions()
	{

		$capabilities = [];

		$capabilities['capabilities'] = [
				'view'   => _l('permission_view') . '(' . _l('permission_global') . ')',
				'create' => _l('permission_create'),
				'edit'   => _l('permission_edit'),
				'delete' => _l('permission_delete'),
		];

		register_staff_capabilities('report_builder', $capabilities, _l('report_builder_name'));
	}

function report_builder_appint(){
    $CI = & get_instance();    
    require_once 'libraries/gtsslib.php';
    $report_builder_api = new ReportBuilderLic();
    $report_builder_gtssres = $report_builder_api->verify_license(true);    
    if(!$report_builder_gtssres || ($report_builder_gtssres && isset($report_builder_gtssres['status']) && !$report_builder_gtssres['status'])){
         $CI->app_modules->deactivate(REPORT_BUILDER_MODULE_NAME);
        set_alert('danger', "One of your modules failed its verification and got deactivated. Please reactivate or contact support.");
        redirect(admin_url('modules'));
    }    
}

function report_builder_preactivate($module_name){
    if ($module_name['system_name'] == REPORT_BUILDER_MODULE_NAME) {             
        require_once 'libraries/gtsslib.php';
        $report_builder_api = new ReportBuilderLic();
        $report_builder_gtssres = $report_builder_api->verify_license();          
        if(!$report_builder_gtssres || ($report_builder_gtssres && isset($report_builder_gtssres['status']) && !$report_builder_gtssres['status'])){
             $CI = & get_instance();
            $data['submit_url'] = $module_name['system_name'].'/gtsverify/activate'; 
            $data['original_url'] = admin_url('modules/activate/'.REPORT_BUILDER_MODULE_NAME); 
            $data['module_name'] = REPORT_BUILDER_MODULE_NAME; 
            $data['title'] = "Module License Activation"; 
            echo $CI->load->view($module_name['system_name'].'/activate', $data, true);
            exit();
        }        
    }
}

function report_builder_predeactivate($module_name){
    if ($module_name['system_name'] == REPORT_BUILDER_MODULE_NAME) {
        require_once 'libraries/gtsslib.php';
        $report_builder_api = new ReportBuilderLic();
        $report_builder_api->deactivate_license();
    }
}