<?php 

final class MagicAI {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI The single instance of the class.
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
	 * @return MagicAI An instance of the class.
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
       
        add_action( 'plugins_loaded', [ $this, 'on_plugins_loaded' ] );

    }

    /**
	 * Load Textdomain
	 *
	 * Load plugin localization files.
	 *
	 * Fired by `init` action hook.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
	public function i18n() {

		load_plugin_textdomain( 'magicai-wp' );

	}

    /**
	 * On Plugins Loaded
	 *
	 * Checks the plugin has loaded, and performs some compatibility checks.
	 * If All checks pass, inits the plugin.
	 *
	 * Fired by `plugins_loaded` action hook.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
	public function on_plugins_loaded() {

		if ( $this->is_compatible() ) {
			$this->init();
		}

	}

    /**
	 * Compatibility Checks
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
	public function is_compatible() {

		return true;

	}

    /**
	 * Initialize the plugin
	 *
	 * Load the files required to run the plugin.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function init() {

        $this->i18n();
		$this->hooks();
        $this->include_files();
        $this->updater();

    } 

	public function hooks() {
		
	}
    
	/**
	 * Includes necessary files for MagicAI plugin functionality.
	 */
    public function include_files() {

		require_once MAGICAI_PATH . 'includes/helper.php';
		require_once MAGICAI_PATH . 'includes/admin/admin-license.php';
		require_once MAGICAI_PATH . 'includes/settings/settings.php';
		require_once MAGICAI_PATH . 'includes/editor/editor.php';
		require_once MAGICAI_PATH . 'includes/post-types/post-types.php';
		require_once MAGICAI_PATH . 'includes/metabox/metabox.php';
		require_once MAGICAI_PATH . 'libs/vendor/autoload.php';
		require_once MAGICAI_PATH . 'includes/admin/admin-init.php';
		require_once MAGICAI_PATH . 'includes/classes/openai.php';
		require_once MAGICAI_PATH . 'includes/classes/stablediffusion.php';
		require_once MAGICAI_PATH . 'includes/classes/aws-s3.php';
		require_once MAGICAI_PATH . 'includes/classes/google-search.php';
		require_once MAGICAI_PATH . 'includes/classes/google-tts.php';
		require_once MAGICAI_PATH . 'includes/classes/logs.php';
		require_once MAGICAI_PATH . 'includes/classes/chatbot.php';
		require_once MAGICAI_PATH . 'includes/classes/rate-limit.php';
		require_once MAGICAI_PATH . 'includes/classes/stats.php';
		require_once MAGICAI_PATH . 'includes/classes/link-crawler.php';
		require_once MAGICAI_PATH . 'includes/hooks.php';
		require_once MAGICAI_PATH . 'includes/shortcodes/shortcodes.php';
		
    }

	/**
	 * Sets up the updater for MagicAI plugin.
	 */
	public function updater() {
		require_once MAGICAI_PATH . 'includes/updater/plugin-update-checker.php';

		$UpdateChecker = YahnisElsts\PluginUpdateChecker\v5\PucFactory::buildUpdateChecker(
			'http://api.liquid-themes.com/magicai-wp/updater.json',
			MAGICAI_PATH . 'magicai-wp.php', // Full path to the main plugin file or functions.php.
			'magicai-wp'
		);
	}

    
} // class
MagicAI::instance();

// Pluging activation hook
function magicai_plugin_activate() { 

    flush_rewrite_rules(); // Removes rewrite rules and then recreate rewrite rules.

}
register_activation_hook( __FILE__, 'magicai_plugin_activate' );

// Pluging deactivation hook
function magicai_plugin_deactivate() {

    flush_rewrite_rules(); // Removes rewrite rules and then recreate rewrite rules.

}
register_deactivation_hook( __FILE__, 'magicai_plugin_deactivate' );
