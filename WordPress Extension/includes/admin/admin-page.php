<?php

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

#[AllowDynamicProperties]
class MagicAI_Admin_Page {

	/**
     * The slug name for the parent menu.
     * 
     * @since 1.0.0
     * 
     * @var string
     */
    public $parent = null;

	/**
     * The capability required for this menu to be displayed to the user.
     * 
     * @since 1.0.0
     * 
     * @var string
     */
    public $capability = 'manage_options';

	/**
     * The icon for this menu.
     *
     * @since 1.0.0
     * 
     * @var string
     */
	public $icon = 'dashicons-art';
	/**
     * The position in the menu order this menu should appear.
     *
     * @since 1.0.0
     * 
     * @var string
     */
    public $position;

	/**
     * Constructor method for initializing the custom admin menu page.
     *
     * This constructor method is responsible for initializing the custom admin menu page
     * and setting up necessary actions and hooks.
     *
     * @since 1.0.0
     *
     * @access public
     */
	public function __construct() {
          
		$priority = -1;
		if ( isset( $this->parent ) && $this->parent ) {
			$priority = intval( $this->position );
		}
		$this->position = 2;
		add_action( 'admin_menu', [ $this, 'register_page' ], $priority );

		if( !isset( $_GET['page'] ) || empty( $_GET['page'] ) || ! $this->id === $_GET['page'] ) {
			return;
		}

		if( method_exists( $this, 'save' ) ) {
			add_action( 'admin_init', [ $this, 'save' ] );
		}
	}

	/**
     * Register a custom admin menu page.
     *
     * This method registers a custom admin menu page in WordPress.
     * 
     * @since 1.0.0
     *
     * @method register_page
     * @return void
     */
	public function register_page() {

		if( ! $this->parent ) {
			add_menu_page(
				$this->page_title,
				$this->menu_title,
				$this->capability,
				$this->id,
				array( $this, 'display' ),
				MAGICAI_URL . 'assets/img/logo.svg',
				$this->position
			);
		}
		else {
			add_submenu_page(
				$this->parent,
				$this->page_title,
				$this->menu_title,
				$this->capability,
				$this->id,
				array( $this, 'display' )
			);
		}
	}

	/**
     * Display the content for the custom admin menu page.
     *
     * This method is responsible for displaying the content for the custom admin menu page.
     *
     * @since 1.0.0
     * 
     * @method display
     */
	public function display() {
		echo 'default';
	}
}
