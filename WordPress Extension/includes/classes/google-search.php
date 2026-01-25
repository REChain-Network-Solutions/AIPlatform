<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

#[AllowDynamicProperties]
class MagicAI_GoogleSearch {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_GoogleSearch The single instance of the class.
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
	 * @return MagicAI_GoogleSearch An instance of the class.
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

		$this->load_options();
		$this->hooks();

    }

    /**
     * Load Options
     *
     * This function loads the MagicAI plugin options from the WordPress database.
     * It retrieves the settings.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function load_options() {

        // Retrieve the MagicAI settings from the database
        $this->api_key = magicai_helper()->get_option( 'google_search_api' );
        $this->cx      = magicai_helper()->get_option( 'google_search_cx' );

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

	}
    
    // TODO: Change Openai request if ai webaccess is enabled.
    // Example 
    // Web search results:
    // %%web_results%%
    // Current date: %%current_date%%
    // Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using <a href="(URL)">[[number]]</a> notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
    // Query: %%original_query%%
    
    function search( $search_query ) {

        if ( empty( $this->api_key ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Custom Search JSON API: Missing API Key', 'magicai-wp' )
            ] );
        }

        if ( empty( $this->cx ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Custom Search JSON API: Missing Search Engine ID', 'magicai-wp' )
            ] );
        }

        $search_query = urlencode( $search_query );
        $url = "https://www.googleapis.com/customsearch/v1?key={$this->$pi_key}&cx={$this->cx}&q=$search_query";
        $response = file_get_contents($url);
        $results = json_decode($response, true);

        $output = '';

        // TODO: Add results count option
        if (isset($results['items'])) {
            $random_keys = array_rand($results['items'], 3);
        
            $i = 1;
            foreach ($random_keys as $key) {
                $item = $results['items'][$key];
                $output .= sprintf( '[%s] %s %s \n URL:%s<br>', ($i), $item['title'], $item['snippet'], $item['link'] );
                $i++;
            }
        }

        if ( ! $output ){
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Google Results not found. Try again or disable the web search feature!', 'magicai-wp' )
            ] );
        }

        return $output;

    }



}
MagicAI_GoogleSearch::instance();
