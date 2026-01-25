<?php

if ( !defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * MagicAI_Admin_Page_Shortcode class for handling the MagicAI admin page.
 *
 * This class extends the MagicAI_Admin_Page class to create a specific admin page for MagicAI.
 *
 * @since 1.0.0
 */
class MagicAI_Admin_Page_Shortcode extends MagicAI_Admin_Page {

	/**
     * Constructor method for initializing the MagicAI admin page.
     *
     * @method __construct
     */
	public function __construct() {

		$this->id = 'magicai-shortcodes';
		$this->page_title = esc_html__( 'Shortcodes', 'magicai-wp' );
		$this->menu_title = esc_html__( 'Shortcodes', 'magicai-wp' );
		$this->parent = 'magicai';
		$this->position = '100';
    
		parent::__construct();
		
		add_action( 'admin_footer', function() {
			?>
			<script type="text/javascript">
				jQuery(document).ready(function($) {
					$('a[href="admin.php?page=magicai-shortcodes"]').click(function(e){
						e.preventDefault();
						window.open('https://magicaidocs-wp.liquid-themes.com/shortcodes/', '_blank')
					});
				});
            </script>
			<?php
		}, 999 );

	}

	/**
     * Display the content for the MagicAI admin page.
     *
     * @method display
     * @return void
     */
	public function display() {
		
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
new MagicAI_Admin_Page_Shortcode;
