<?php

if ( !defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * MagicAI_Admin_Page_Settings class for handling the MagicAI admin page.
 *
 * This class extends the MagicAI_Admin_Page class to create a specific admin page for MagicAI.
 *
 * @since 1.0.0
 */
class MagicAI_Admin_Page_Settings extends MagicAI_Admin_Page {

	/**
     * Constructor method for initializing the MagicAI admin page.
     *
     * @method __construct
     */
	public function __construct() {

		$this->id = 'magicai-settings';
		$this->page_title = esc_html__( 'Settings', 'magicai-wp' );
		$this->menu_title = esc_html__( 'Settings', 'magicai-wp' );
		$this->parent = 'magicai';
		$this->position = '90';
        
		parent::__construct();
	}

	/**
     * Display the content for the MagicAI admin page.
     *
     * @method display
     * @return void
     */
	public function display() {
		MagicAI_Main_Settings::instance()->render_options_page();
	}

	/**
     * Save method for handling data saving on the MagicAI admin page.
     *
     * @method save
     * @return void
     */
	public function save() {

	}
}
new MagicAI_Admin_Page_Settings;
