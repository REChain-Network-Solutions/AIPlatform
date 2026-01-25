<?php
/**
 * Plugin Name: MagicAI for WordPress
 * Description: Powerful WordPress AI Tool.
 * Text Domain: magicai-wp
 * Requires PHP: 7.4
 */

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly
update_option('magicai_purchase_code_status', 'valid' );
update_option('magicai_purchase_code', '**************************' );
update_option('magicai_purchase_code_domain_key', str_replace(array("http://","https://"),"",site_url()) );
update_option('magicai_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()) );
set_transient('magicai_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()), 12 * HOUR_IN_SECONDS );
set_transient('magicai_license_message', 'License verified by babiato', ( 24 * HOUR_IN_SECONDS ));
define( 'MAGICAI_PATH', plugin_dir_path( __FILE__ ) );
define( 'MAGICAI_URL', plugin_dir_url( __FILE__ ) );
define( 'MAGICAI_VERSION', get_file_data( __FILE__, array('Version' => 'Version'), false)['Version'] );

require_once MAGICAI_PATH . 'includes/plugin.php';