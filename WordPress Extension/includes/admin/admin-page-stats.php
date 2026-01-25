<?php

if ( !defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * MagicAI_Admin_Page_Logs class for handling the MagicAI admin page.
 *
 * This class extends the MagicAI_Admin_Page class to create a specific admin page for MagicAI.
 *
 * @since 1.4
 */
class MagicAI_Admin_Page_Statistics extends MagicAI_Admin_Page {

	/**
     * Constructor method for initializing the MagicAI admin page.
     *
     * @method __construct
     */
	public function __construct() {

		$this->id = 'magicai-stats';
		$this->page_title = esc_html__( 'Statistics', 'magicai-wp' );
		$this->menu_title = esc_html__( 'Statistics', 'magicai-wp' );
		$this->parent = 'magicai';
		$this->position = '110';
        
		parent::__construct();
	}

	/**
     * Display the content for the MagicAI admin page.
     *
     * @method display
     * @return void
     */
	public function display() {

		include_once __DIR__ . '/views/stats.php';

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
new MagicAI_Admin_Page_Statistics;
