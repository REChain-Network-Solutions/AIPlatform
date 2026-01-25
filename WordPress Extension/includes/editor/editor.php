<?php

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

class MagicAI_Editor {
	
	public function __construct() {
		add_action( 'enqueue_block_editor_assets', [ $this, 'enqueue_block_editor_assets' ], 9999 );
        add_action( 'admin_init', [ $this, 'magicai_tinymce_button' ] );
	}

    /**
     * Enqueue block editor assets.
     */
    function enqueue_block_editor_assets() {
        wp_enqueue_script(
            'magicai-block-editor',
            MAGICAI_URL . 'assets/js/editor/block.js',
            array( 'wp-blocks', 'wp-dom' )
        );
    }

    /**
     * Add a TinyMCE button.
     * https://developer.wordpress.org/reference/hooks/mce_external_plugins/#user-contributed-notes
     */
    function magicai_tinymce_button() {
        if ( current_user_can( 'edit_posts' ) && current_user_can( 'edit_pages' ) ) {
            add_filter( 'mce_buttons', [$this, 'magicai_register_tinymce_button'] );
            add_filter( 'mce_external_plugins', [$this,'magicai_add_tinymce_button'] );
        }
    }

    /**
     * Register a TinyMCE button.
     */
    function magicai_register_tinymce_button( $buttons ) {
        array_push( $buttons, "magicai_button" );
        return $buttons;
    }

    /**
     * Add a TinyMCE button script.
     */
    function magicai_add_tinymce_button( $plugin_array ) {
        $plugin_array['magicai_classic_button_script'] = MAGICAI_URL . 'assets/js/editor/classic.js';
        return $plugin_array;
    }

}

new MagicAI_Editor();
