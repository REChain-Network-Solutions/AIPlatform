<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

class MagicAI_Admin {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Admin The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_Admin An instance of the class.
	 */
	public static function instance() {

		if ( is_null( self::$_instance ) ) {
			self::$_instance = new self();
		}
		return self::$_instance;
    }

    /**
	 * Constructor
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function __construct() {
       
        $this->init();

    }

    /**
	 * Initialize
	 *
	 * Load the files required to run the plugin.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function init() {

		$this->hooks();
		$this->include_files();

    }

    /**
     * Hooks Registration
     *
     * This function registers WordPress action and filter hooks for the MagicAI plugin.
     * It allows you to define custom actions and filters to extend or modify the plugin's functionality.
     *
     * @since 1.0.0
     *
     * @access public
     */
	public function hooks() {

        add_action( 'admin_enqueue_scripts', [ $this, 'admin_enqueue_scripts' ] );
        add_action( 'wp_enqueue_scripts', [ $this, 'frontend_enqueue_scripts' ] );
        
        
        add_action( 'admin_bar_menu', [ $this, 'admin_bar_item' ], 200 );
        add_action('admin_menu', [ $this, 'fix_parent_menu' ], 999);

	}

    /**
     * Include Required Files
     *
     * This function includes the necessary PHP files and dependencies for the MagicAI plugin to function correctly.
     * It ensures that all required files are loaded for proper execution of the plugin's features.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function include_files() {

        // Admin Pages
        require_once MAGICAI_PATH . 'includes/admin/admin-page.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-dashboard.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-generators.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-chat.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-chat-pdf.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-vision.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-speech-to-text.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-voiceover.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-chatbot.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-settings.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-shortcodes.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-stats.php';
        require_once MAGICAI_PATH . 'includes/admin/admin-page-logs.php';
        
    }

    /**
     * Admin Enqueue Scripts
     *
     * This function is responsible for enqueuing JavaScript and CSS files for the admin section of the MagicAI plugin.
     *
     * @since 1.0.0
     *
     * @param string $hook The current admin page hook.
     *
     * @access public
     */
    function admin_enqueue_scripts( $hook ) {

		wp_enqueue_script( 'notyf', MAGICAI_URL . 'assets/vendors/notfy/notyf.min.js', [], '3.0', false );
		wp_enqueue_style( 'notyf', MAGICAI_URL . 'assets/vendors/notfy/notyf.min.css', [], '3.0' );

        if ( $hook == 'magicai_page_magicai-settings' ){
            wp_enqueue_script( 'Sortable', MAGICAI_URL . 'assets/vendors/Sortable.min.js', [], '1.5.1' );
        }

		wp_enqueue_style( 'magicai-vars', MAGICAI_URL . 'assets/css/vars.css', [], MAGICAI_VERSION );
		wp_enqueue_script( 'magicai-admin-settings', MAGICAI_URL . 'assets/js/admin-settings.js', [], MAGICAI_VERSION, false );
		wp_enqueue_style( 'magicai-admin-settings', MAGICAI_URL . 'assets/css/admin-settings.css', [], MAGICAI_VERSION );
        
        // chatbot
		wp_enqueue_style( 'magicai-chatbot', MAGICAI_URL . 'assets/css/chatbot.css', ['magicai-vars'], MAGICAI_VERSION );
		wp_enqueue_script( 'magicai-chatbot', MAGICAI_URL . 'assets/js/chatbot.js', ['magicai-admin-settings'], MAGICAI_VERSION, false );
        if ( get_post_type() == 'magicai-chatbot' || $hook == 'magicai_page_magicai-chatbot' ) {
            wp_enqueue_script( 'magicai-chatbot-settings', MAGICAI_URL . 'assets/js/chatbot-settings.js', ['magicai-admin-settings'], MAGICAI_VERSION, false );
        }
		
		// jquery confirm
		wp_enqueue_script( 'magicai-jquery-confirm', MAGICAI_URL . 'assets/vendors/jquery-confirm.min.js', ['jquery'], '3.3.4', false );
		wp_enqueue_style( 'magicai-jquery-confirm', MAGICAI_URL . 'assets/vendors/jquery-confirm.min.css', [], '3.3.4' );

        // code highlighter
        if ( in_array( $hook, [ 'magicai_page_magicai-generators', 'magicai_page_magicai-speech-to-text', 'magicai_page_magicai-vision', 'magicai_page_magicai-chat-pdf' ] ) || get_post_type() == 'magicai-chatbot' ) { // Enqueue just for this page
            wp_enqueue_editor();
            wp_enqueue_media();
            wp_enqueue_script( 'prism', MAGICAI_URL . 'assets/vendors/prism/prism.js', [], '1.29', true );
		    wp_enqueue_style( 'prism', MAGICAI_URL . 'assets/vendors/prism/prism.css', [], '1.29' );
        }

        if ( $hook == 'magicai_page_magicai-voiceover' || get_post_type() == 'magicai-documents' ) {
            wp_enqueue_script( 'magicai-voiceover', MAGICAI_URL . 'assets/js/voiceover.js', ['magicai-admin-settings'], MAGICAI_VERSION, false );
            wp_enqueue_script( 'wavesurfer', MAGICAI_URL . 'assets/vendors/wavesurfer.js', ['magicai-voiceover'], '6.6.4', false );
        }

        if ( in_array( $hook, [ 'magicai_page_magicai-stats' ] ) ) { // Enqueue just for this page
            wp_enqueue_script( 'chart', MAGICAI_URL . 'assets/vendors/chart.umd.min.js', [], '4.4.1', true );
            wp_enqueue_script( 'magicai-stats', MAGICAI_URL . 'assets/js/stats.js', ['chart'], MAGICAI_VERSION, true );
        }


	}

    /**
     * Enqueue Scripts
     *
     * This function is responsible for enqueuing JavaScript and CSS files for the frontend section of the MagicAI plugin.
     *
     * @since 1.0.0
     *
     * @access public
     */
    function frontend_enqueue_scripts() {

        wp_enqueue_script( 'notyf', MAGICAI_URL . 'assets/vendors/notfy/notyf.min.js', [], '3.0', false );
        wp_enqueue_style( 'notyf', MAGICAI_URL . 'assets/vendors/notfy/notyf.min.css', [], '3.0' );

        wp_enqueue_style( 'magicai-vars', MAGICAI_URL . 'assets/css/vars.css', [], MAGICAI_VERSION );
		wp_enqueue_script( 'magicai-admin-settings', MAGICAI_URL . 'assets/js/admin-settings.js', [], MAGICAI_VERSION, false );
		wp_enqueue_style( 'magicai-admin-settings', MAGICAI_URL . 'assets/css/admin-settings.css', [], MAGICAI_VERSION );

        $page_content = get_the_content();
        if ( 
            has_shortcode( $page_content, 'magicai-generator' ) ||
            has_shortcode( $page_content, 'magicai-image-generator' ) ||
            has_shortcode( $page_content, 'magicai-chat' ) ||
            has_shortcode( $page_content, 'magicai-logs' ) ||
            has_shortcode( $page_content, 'magicai-documents' )
         ) {
            wp_enqueue_editor();
            wp_enqueue_media();
        }
        wp_enqueue_script( 'prism', MAGICAI_URL . 'assets/vendors/prism/prism.js', [], '1.29', true );
        wp_enqueue_style( 'prism', MAGICAI_URL . 'assets/vendors/prism/prism.css', [], '1.29' );

        $options = get_option('magicai_chatbot_settings', array() );
        if ( isset( $options['status'] ) && $options['status'] != 'disabled' ) {
            wp_enqueue_style( 'magicai-vars', MAGICAI_URL . 'assets/css/vars.css', [], MAGICAI_VERSION );
            wp_enqueue_style( 'magicai-chatbot', MAGICAI_URL . 'assets/css/chatbot.css', ['magicai-vars'], MAGICAI_VERSION );
            wp_enqueue_script( 'magicai-chatbot', MAGICAI_URL . 'assets/js/chatbot.js', ['jquery'], MAGICAI_VERSION, false );
        }

		wp_enqueue_script( 'magicai-jquery-confirm', MAGICAI_URL . 'assets/vendors/jquery-confirm.min.js', ['jquery'], '3.3.4', false );
		wp_enqueue_style( 'magicai-jquery-confirm', MAGICAI_URL . 'assets/vendors/jquery-confirm.min.css', [], '3.3.4' );


	}
    
    function admin_bar_item( $admin_bar ) {
        if ( ! current_user_can( 'manage_options' ) ) {
            return;
        }

        $title = sprintf( '<div class="fix-menu"><img src="%s" style="width:16px;height:16px;"> MagicAI</div>', MAGICAI_URL . 'assets/img/logo.svg' );

        // Generators
        $admin_bar->add_menu( array(
            'id'    => 'magicai',
            'parent' => null, // top-secondary : top right
            'title' => $title,
            'href'  => admin_url('admin.php?page=magicai'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators',
            'parent' => 'magicai',
            'title' => esc_html__('Post', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators-rss-post-generator',
            'parent' => 'magicai',
            'title' => esc_html__('RSS to Post', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators#tab-rss-post-generator'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators-yt-post-generator',
            'parent' => 'magicai',
            'title' => esc_html__('AI Youtube', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators#tab-yt-post-generator'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators-product-generator',
            'parent' => 'magicai',
            'title' => esc_html__('Product', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators#tab-product-generator'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators-custom-generator',
            'parent' => 'magicai',
            'title' => esc_html__('Custom', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators#tab-custom-generator'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators-code-generator',
            'parent' => 'magicai',
            'title' => esc_html__('Code', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators#tab-custom-generator'),
        ) );
        
        $admin_bar->add_menu( array(
            'id'    => 'generators-image-generator',
            'parent' => 'magicai',
            'title' => esc_html__('Image', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-generators#tab-image-generator'),
        ) );

        // Other
        $admin_bar->add_menu( array(
            'id'    => 'chat',
            'parent' => 'magicai',
            'title' => esc_html__('Chat', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-chat'),
        ) );

        $admin_bar->add_menu( array(
            'id'    => 'vision',
            'parent' => 'magicai',
            'title' => esc_html__('Vision', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-vision'),
        ) );
  
        $admin_bar->add_menu( array(
            'id'    => 'speech-to-text',
            'parent' => 'magicai',
            'title' => esc_html__('Speech to Text', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-speech-to-text'),
        ) );
  
        $admin_bar->add_menu( array(
            'id'    => 'voiceover',
            'parent' => 'magicai',
            'title' => esc_html__('Voiceover', 'magicai-wp'), 
            'href'  => admin_url('admin.php?page=magicai-voiceover'),
        ) );
        

    }

    /**
     * Modifies the label of the 'Dashboard' submenu item under the 'magicai' menu.
     * This function is intended for users with 'edit_theme_options' capability.
     * It changes the label to a localized string provided by the 'magicai-wp' domain.
     */
	public function fix_parent_menu() {

		if (!current_user_can('edit_theme_options')) {
			return;
		}

		global $submenu;

		$submenu['magicai'][0][0] = esc_html__( 'Dashboard', 'magicai-wp' );

	}

}
MagicAI_Admin::instance();
